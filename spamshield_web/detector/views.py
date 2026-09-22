"""Views for the SpamShield detector application."""

from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .models import PredictionLog
from .services import (
    batch_predict,
    explain_prediction,
    get_model_metadata,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def index(request):
    """Main live message classifier view."""
    result = None
    input_text = ""

    if request.method == "POST":
        input_text = request.POST.get("message_text", "").strip()
        if input_text:
            result = explain_prediction(input_text)

            # Persist to database log
            try:
                PredictionLog.objects.create(
                    text=input_text,
                    predicted_label=result["label"],
                    spam_probability=result["spam_probability"],
                    risk_level=result["risk_level"],
                    top_contributing_tokens=result["all_contributions"][:6],
                    inference_time_ms=result["inference_time_ms"],
                )
            except Exception as e:
                print(f"[Warning] Failed to log prediction: {e}")

    # Presets for instant testing in viva
    presets = [
        {
            "title": "❤️ Aunt May Dinner SMS (Clean Ham)",
            "badge": "Safe Ham",
            "badge_color": "success",
            "text": "Hey Peter, don't forget to pick up eggs on your way back from the library. Dinner is at 7:30! Love you.",
        },
        {
            "title": "📰 Daily Bugle Cash Draw (Classic Spam)",
            "badge": "Threat Spam",
            "badge_color": "danger",
            "text": "URGENT! You have won £50000 cash prize or a holiday voucher! Call 08718726270 NOW to claim your guaranteed reward. T&C apply.",
        },
        {
            "title": "⚡ Oscorp Security Breach (Critical Phishing)",
            "badge": "Phishing Alert",
            "badge_color": "danger",
            "text": "ALERT: Your Oscorp bank account #4928 is temporarily restricted due to suspicious logins. Verify immediately at http://secure-bank-login.xyz/verify",
        },
        {
            "title": "🎓 Peter & Ned Leeds Project Chat (Clean Ham)",
            "badge": "Safe Ham",
            "badge_color": "success",
            "text": "Hey man, did you finish training the Logistic Regression model for our physics lab? Bring the notebook tomorrow!",
        },
    ]

    metadata = get_model_metadata()
    context = {
        "result": result,
        "input_text": input_text,
        "presets": presets,
        "metadata": metadata,
        "metrics": metadata.get("sklearn_metrics_tuned", metadata.get("sklearn_metrics", {})),
    }
    return render(request, "detector/index.html", context)


def batch_view(request):
    """Batch message analyzer view supporting multi-line text and CSV uploads."""
    results = None
    summary = None

    if request.method == "POST":
        messages = []

        # 1. Check CSV upload
        csv_file = request.FILES.get("csv_file")
        if csv_file:
            try:
                decoded_file = csv_file.read().decode("utf-8", errors="ignore")
                reader = csv.reader(io.StringIO(decoded_file))
                header = next(reader, None)
                text_col_idx = 0
                if header:
                    for i, col in enumerate(header):
                        if "text" in col.lower() or "message" in col.lower() or "sms" in col.lower():
                            text_col_idx = i
                            break
                    for row in reader:
                        if row and len(row) > text_col_idx and row[text_col_idx].strip():
                            messages.append(row[text_col_idx].strip())
            except Exception as e:
                messages = []
                print(f"[Batch] CSV parsing error: {e}")

        # 2. Check multi-line text input
        batch_text = request.POST.get("batch_text", "").strip()
        if batch_text and not messages:
            messages = [line.strip() for line in batch_text.splitlines() if line.strip()]

        # Limit to 500 messages per request to maintain responsive server
        messages = messages[:500]

        if messages:
            predictions = batch_predict(messages)
            spam_count = sum(1 for p in predictions if p["is_spam"])
            ham_count = len(predictions) - spam_count
            spam_percentage = (spam_count / len(predictions)) * 100 if predictions else 0.0

            results = predictions
            summary = {
                "total": len(predictions),
                "spam_count": spam_count,
                "ham_count": ham_count,
                "spam_percentage": round(spam_percentage, 1),
            }

    context = {
        "results": results,
        "summary": summary,
    }
    return render(request, "detector/batch.html", context)


def export_batch_csv(request):
    """Export the most recent batch classification results to a CSV file."""
    if request.method == "POST":
        data_json = request.POST.get("results_json", "[]")
        try:
            records = json.loads(data_json)
        except Exception:
            records = []

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="spamshield_classified_batch.csv"'

        writer = csv.writer(response)
        writer.writerow(["Index", "Classification", "Spam_Probability", "Risk_Level", "Message_Text"])

        for i, r in enumerate(records, start=1):
            writer.writerow([
                i,
                r.get("label", ""),
                f"{r.get('spam_probability', 0.0):.4f}",
                r.get("risk_level", ""),
                r.get("text", ""),
            ])

        return response
    return redirect("batch")


def dashboard_view(request):
    """Academic model metrics, visual confusion matrices, and feature importance."""
    metadata = get_model_metadata()

    # Load top spam and ham coefficients from pipeline if available
    top_spam = metadata.get("top_spam_indicators", [])
    top_ham = metadata.get("top_ham_indicators", [])

    context = {
        "metadata": metadata,
        "scratch_metrics": metadata.get("scratch_metrics", {}),
        "sklearn_metrics": metadata.get("sklearn_metrics", {}),
        "sklearn_metrics_tuned": metadata.get("sklearn_metrics_tuned", {}),
        "optimal_threshold": metadata.get("optimal_threshold", 0.50),
        "top_spam": top_spam,
        "top_ham": top_ham,
    }
    return render(request, "detector/dashboard.html", context)


def history_view(request):
    """Prediction audit log and history table."""
    logs = PredictionLog.objects.all()[:100]
    total_logged = PredictionLog.objects.count()
    spam_logged = PredictionLog.objects.filter(predicted_label="SPAM").count()
    ham_logged = total_logged - spam_logged

    context = {
        "logs": logs,
        "total_logged": total_logged,
        "spam_logged": spam_logged,
        "ham_logged": ham_logged,
    }
    return render(request, "detector/history.html", context)


def clear_history(request):
    """Clear all stored prediction logs."""
    if request.method == "POST":
        PredictionLog.objects.all().delete()
    return redirect("history")


# --- REST API ENDPOINTS ---

@csrf_exempt
def api_predict(request):
    """REST API endpoint for real-time single message classification.
    
    Accepts: POST application/json {"text": "..."}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        text = body.get("text", "").strip()
    except Exception:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    if not text:
        return JsonResponse({"error": "Field 'text' is required."}, status=400)

    result = explain_prediction(text)

    # Persist to database log for audit trail
    try:
        PredictionLog.objects.create(
            text=text,
            predicted_label=result["label"],
            spam_probability=result["spam_probability"],
            risk_level=result["risk_level"],
            top_contributing_tokens=result["all_contributions"][:6],
            inference_time_ms=result["inference_time_ms"],
        )
    except Exception as e:
        print(f"[Warning] Failed to log API prediction: {e}")

    return JsonResponse({
        "success": True,
        "text": text,
        "label": result["label"],
        "is_spam": result["is_spam"],
        "spam_probability": result["spam_probability"],
        "ham_probability": result["ham_probability"],
        "spam_percentage": result.get("spam_percentage", round(result["spam_probability"] * 100, 1)),
        "ham_percentage": result.get("ham_percentage", round(result["ham_probability"] * 100, 1)),
        "risk_level": result["risk_level"],
        "inference_time_ms": result["inference_time_ms"],
        "optimal_threshold": result.get("optimal_threshold", 0.35),
        "spam_drivers": result.get("spam_drivers", []),
        "ham_drivers": result.get("ham_drivers", []),
        "top_signals": result["all_contributions"][:6],
    })


@csrf_exempt
def api_batch_predict(request):
    """REST API endpoint for batch classification."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed. Use POST."}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        texts = body.get("texts", [])
    except Exception:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    if not isinstance(texts, list) or not texts:
        return JsonResponse({"error": "Field 'texts' must be a non-empty list of strings."}, status=400)

    # Cap batch at 200 items per call
    texts = [str(t) for t in texts[:200]]
    results = batch_predict(texts)

    return JsonResponse({
        "success": True,
        "count": len(results),
        "results": results,
    })

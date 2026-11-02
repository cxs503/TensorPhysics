import sys, json
sys.path.insert(0, "src")
from tensorphysics.platform_reporting import register_run_outcome
from tensorphysics.platform_registry import TensorPlatformRunRegistry
try:
    reg = TensorPlatformRunRegistry("http://exa mple.invalid", "tok", 1)
    try:
        out = register_run_outcome(run_id="r", status="succeeded", validation_status="passed",
                                   manifest={}, registry=reg)
        print("outcome:", out)
    except Exception as e:
        import http.client
        print("ESCAPED:", type(e).__module__ + "." + type(e).__name__,
              "| is ValueError:", isinstance(e, ValueError),
              "| is OSError:", isinstance(e, OSError),
              "| is HTTPException:", isinstance(e, http.client.HTTPException))
except Exception as e:
    print("ctor rejected:", type(e).__name__, e)

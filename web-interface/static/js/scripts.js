
document.addEventListener("DOMContentLoaded", function() {
  const loadBtn = document.getElementById("loadSample");
  if (!loadBtn) return;

  loadBtn.addEventListener("click", () => {
    // Example sample values (replace with realistic numbers):
    document.getElementById("close").value = 150.00;
    document.getElementById("volume").value = 1200000;
    document.getElementById("ret_1d").value = 0.0123;
    document.getElementById("ret_2d").value = -0.0056;
    document.getElementById("momentum_5d").value = 1.025;
    document.getElementById("ma_5d").value = 148.75;
    document.getElementById("vol_5d").value = 2.34;
    document.getElementById("accel").value = 0.0012;
    document.getElementById("rsi_14").value = 58.39;
  });
});

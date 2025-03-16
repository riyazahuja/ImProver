@[fun_prop, measurability]
protected theorem measurable (L : E →L[𝕜] F) : Measurable L :=
  L.continuous.measurable


@[fun_prop]
theorem measurable_comp (L : E →L[𝕜] F) {φ : α → E} (φ_meas : Measurable φ) :
    Measurable fun a : α => L (φ a) :=
  L.measurable.comp φ_meas


instance instMeasurableSpace : MeasurableSpace (E →L[𝕜] F) :=
  borel _


instance instBorelSpace : BorelSpace (E →L[𝕜] F) :=
  ⟨rfl⟩


@[fun_prop, measurability]
theorem measurable_apply [MeasurableSpace F] [BorelSpace F] (x : E) :
    Measurable fun f : E →L[𝕜] F => f x :=
  (apply 𝕜 F x).continuous.measurable


@[measurability]
theorem measurable_apply' [MeasurableSpace E] [OpensMeasurableSpace E] [MeasurableSpace F]
    [BorelSpace F] : Measurable fun (x : E) (f : E →L[𝕜] F) => f x :=
  measurable_pi_lambda _ fun f => f.measurable


@[measurability]
theorem measurable_coe [MeasurableSpace F] [BorelSpace F] :
    Measurable fun (f : E →L[𝕜] F) (x : E) => f x :=
  measurable_pi_lambda _ measurable_apply


@[fun_prop, measurability]
theorem Measurable.apply_continuousLinearMap {φ : α → F →L[𝕜] E} (hφ : Measurable φ) (v : F) :
    Measurable fun a => φ a v :=
  (ContinuousLinearMap.apply 𝕜 E v).measurable.comp hφ


@[measurability]
theorem AEMeasurable.apply_continuousLinearMap {φ : α → F →L[𝕜] E} {μ : Measure α}
    (hφ : AEMeasurable φ μ) (v : F) : AEMeasurable (fun a => φ a v) μ :=
  (ContinuousLinearMap.apply 𝕜 E v).measurable.comp_aemeasurable hφ


theorem measurable_smul_const {f : α → 𝕜} {c : E} (hc : c ≠ 0) :
    (Measurable fun x => f x • c) ↔ Measurable f :=
  (isClosedEmbedding_smul_left hc).measurableEmbedding.measurable_comp_iff


theorem aemeasurable_smul_const {f : α → 𝕜} {μ : Measure α} {c : E} (hc : c ≠ 0) :
    AEMeasurable (fun x => f x • c) μ ↔ AEMeasurable f μ :=
  (isClosedEmbedding_smul_left hc).measurableEmbedding.aemeasurable_comp_iff



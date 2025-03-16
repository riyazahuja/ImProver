/-- The Minkowski functional for vector spaces over normed fields.
Given a set `s` in a vector space over a normed field `𝕜`,
`egauge s` is the functional which sends `x : E`
to the infimum of `‖c‖₊` over `c` such that `x` belongs to `s` scaled by `c`.

The definition only requires `𝕜` to have a `NNNorm` instance
and `(· • ·) : 𝕜 → E → E` to be defined.
This way the definition applies, e.g., to `𝕜 = ℝ≥0`.
For `𝕜 = ℝ≥0`, the function is equal (up to conversion to `ℝ`)
to the usual Minkowski functional defined in `gauge`. -/
noncomputable def egauge (𝕜 : Type*) [NNNorm 𝕜] {E : Type*} [SMul 𝕜 E] (s : Set E) (x : E) : ℝ≥0∞ :=
  ⨅ (c : 𝕜) (_ : x ∈ c • s), ‖c‖₊


@[mono, gcongr]
lemma egauge_anti (h : s ⊆ t) (x : E) : egauge 𝕜 t x ≤ egauge 𝕜 s x :=
  iInf_mono fun _c ↦ iInf_mono' fun hc ↦ ⟨smul_set_mono h hc, le_rfl⟩


                                                            /-
                                                              𝕜 : Type u_1
                                                              inst✝¹ : NNNorm 𝕜
                                                              E : Type u_2
                                                              inst✝ : SMul 𝕜 E
                                                              x : E
                                                              ⊢ Eq (egauge 𝕜 EmptyCollection.emptyCollection x) Top.top
                                                            -/
@[simp] lemma egauge_empty (x : E) : egauge 𝕜 ∅ x = ∞ := by simp [egauge]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma egauge_le_of_mem_smul (h : x ∈ c • s) : egauge 𝕜 s x ≤ ‖c‖₊ := iInf₂_le c h


lemma le_egauge_iff : r ≤ egauge 𝕜 s x ↔ ∀ c : 𝕜, x ∈ c • s → r ≤ ‖c‖₊ := le_iInf₂_iff


                                                                  /-
                                                                    𝕜 : Type u_1
                                                                    inst✝¹ : NNNorm 𝕜
                                                                    E : Type u_2
                                                                    inst✝ : SMul 𝕜 E
                                                                    s : Set E
                                                                    x : E
                                                                    ⊢ Iff (Eq (egauge 𝕜 s x) Top.top) (∀ (c : 𝕜), Not (Membership.mem (HSMul.hSMul …
                                                                  -/
lemma egauge_eq_top : egauge 𝕜 s x = ∞ ↔ ∀ c : 𝕜, x ∉ c • s := by simp [egauge]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma egauge_lt_iff : egauge 𝕜 s x < r ↔ ∃ c : 𝕜, x ∈ c • s ∧ ‖c‖₊ < r := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NNNorm 𝕜
    E : Type u_2
    inst✝ : SMul 𝕜 E
    s : Set E
    x : E
    r : ENNReal
    ⊢ Iff (LT.lt (egauge 𝕜 s x) r) (Exists fun c => And (Membership.mem (HSMul.hSM …
  -/
  simp [egauge, iInf_lt_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma egauge_zero_left_eq_top : egauge 𝕜 0 x = ∞ ↔ x ≠ 0 := by
  /-
    𝕜 : Type u_1
    inst✝³ : NNNorm 𝕜
    inst✝² : Nonempty 𝕜
    E : Type u_2
    inst✝¹ : Zero E
    inst✝ : SMulZeroClass 𝕜 E
    x : E
    ⊢ Iff (Eq (egauge 𝕜 0 x) Top.top) (Ne x 0)
  -/
  simp [egauge_eq_top]
  /-
    🎉 no goals
  -/


@[simp] alias ⟨_, egauge_zero_left⟩ := egauge_zero_left_eq_top


/-- If `c • x ∈ s` and `c ≠ 0`, then `egauge 𝕜 s x` is at most `((‖c‖₊⁻¹ : ℝ≥0) : ℝ≥0∞).

See also `egauge_le_of_smul_mem`. -/
lemma egauge_le_of_smul_mem_of_ne (h : c • x ∈ s) (hc : c ≠ 0) :
    egauge 𝕜 s x ≤ ↑(‖c‖₊⁻¹ : ℝ≥0) := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    h : Membership.mem s (HSMul.hSMul c x)
    hc : Ne c 0
    ⊢ LE.le (egauge 𝕜 s x) ↑(Inv.inv (NNNorm.nnnorm c))
  -/
  rw [← nnnorm_inv]
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    h : Membership.mem s (HSMul.hSMul c x)
    hc : Ne c 0
    ⊢ LE.le (egauge 𝕜 s x) ↑(NNNorm.nnnorm (Inv.inv c))
  -/
  exact egauge_le_of_mem_smul <| (mem_inv_smul_set_iff₀ hc _ _).2 h
  /-
    🎉 no goals
  -/


/-- If `c • x ∈ s`, then `egauge 𝕜 s x` is at most `(‖c‖₊ : ℝ≥0∞)⁻¹.

See also `egauge_le_of_smul_mem_of_ne`. -/
lemma egauge_le_of_smul_mem (h : c • x ∈ s) : egauge 𝕜 s x ≤ (‖c‖₊ : ℝ≥0∞)⁻¹ := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    h : Membership.mem s (HSMul.hSMul c x)
    ⊢ LE.le (egauge 𝕜 s x) (Inv.inv ↑(NNNorm.nnnorm c))
  -/
  rcases eq_or_ne c 0 with rfl | hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      h : Membership.mem s (HSMul.hSMul 0 x)
      ⊢ LE.le (egauge 𝕜 s x) (Inv.inv ↑(NNNorm.nnnorm 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      x : E
      h : Membership.mem s (HSMul.hSMul c x)
      hc : Ne c 0
      ⊢ LE.le (egauge 𝕜 s x) (Inv.inv ↑(NNNorm.nnnorm c))
    -/
  · exact (egauge_le_of_smul_mem_of_ne h hc).trans ENNReal.coe_inv_le
    /-
      🎉 no goals
    -/


lemma mem_of_egauge_lt_one (hs : Balanced 𝕜 s) (hx : egauge 𝕜 s x < 1) : x ∈ s :=
  let ⟨c, hxc, hc⟩ := egauge_lt_iff.1 hx
  hs c (mod_cast hc.le) hxc


lemma egauge_eq_zero_iff : egauge 𝕜 s x = 0 ↔ ∃ᶠ c : 𝕜 in 𝓝 0, x ∈ c • s := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (Eq (egauge 𝕜 s x) 0) (Filter.Frequently (fun c => Membership.mem (HSMul …
  -/
  refine (iInf₂_eq_bot _).trans ?_
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.l …
  -/
  rw [(nhds_basis_uniformity uniformity_basis_edist).frequently_iff]
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    ⊢ Iff (∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.l …
  -/
  simp [and_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma egauge_zero_right (hs : s.Nonempty) : egauge 𝕜 s 0 = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : s.Nonempty
    ⊢ Eq (egauge 𝕜 s 0) 0
  -/
  have : 0 ∈ (0 : 𝕜) • s := by simp [zero_smul_set hs]
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : s.Nonempty
    this : Membership.mem (HSMul.hSMul 0 s) 0
    ⊢ Eq (egauge 𝕜 s 0) 0
  -/
  simpa using egauge_le_of_mem_smul this
  /-
    🎉 no goals
  -/


                                                          /-
                                                            𝕜 : Type u_1
                                                            inst✝² : NormedDivisionRing 𝕜
                                                            E : Type u_2
                                                            inst✝¹ : AddCommGroup E
                                                            inst✝ : Module 𝕜 E
                                                            ⊢ Eq (egauge 𝕜 0 0) 0
                                                          -/
lemma egauge_zero_zero : egauge 𝕜 (0 : Set E) 0 = 0 := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma egauge_le_one (h : x ∈ s) : egauge 𝕜 s x ≤ 1 := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    h : Membership.mem s x
    ⊢ LE.le (egauge 𝕜 s x) 1
  -/
  rw [← one_smul 𝕜 s] at h
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    h : Membership.mem (HSMul.hSMul 1 s) x
    ⊢ LE.le (egauge 𝕜 s x) 1
  -/
  simpa using egauge_le_of_mem_smul h
  /-
    🎉 no goals
  -/


lemma le_egauge_smul_left (c : 𝕜) (s : Set E) (x : E) :
    egauge 𝕜 s x / ‖c‖₊ ≤ egauge 𝕜 (c • s) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    ⊢ LE.le (HDiv.hDiv (egauge 𝕜 s x) ↑(NNNorm.nnnorm c)) (egauge 𝕜 (HSMul.hSMul c …
  -/
  simp_rw [le_egauge_iff, smul_smul]
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    ⊢ ∀ (c_1 : 𝕜), Membership.mem (HSMul.hSMul (HMul.hMul c_1 c) s) x → LE.le (HDi …
  -/
  rintro a ⟨x, hx, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    a : 𝕜
    x : E
    hx : Membership.mem s x
    ⊢ LE.le (HDiv.hDiv (egauge 𝕜 s ((fun x => HSMul.hSMul (HMul.hMul a c) x) x)) ↑ …
  -/
  apply ENNReal.div_le_of_le_mul
  /-
    case intro.intro.h
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    a : 𝕜
    x : E
    hx : Membership.mem s x
    ⊢ LE.le (egauge 𝕜 s ((fun x => HSMul.hSMul (HMul.hMul a c) x) x)) (HMul.hMul ↑ …
  -/
  rw [← ENNReal.coe_mul, ← nnnorm_mul]
  /-
    case intro.intro.h
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    a : 𝕜
    x : E
    hx : Membership.mem s x
    ⊢ LE.le (egauge 𝕜 s ((fun x => HSMul.hSMul (HMul.hMul a c) x) x)) ↑(NNNorm.nnn …
  -/
  exact egauge_le_of_mem_smul <| smul_mem_smul_set hx
  /-
    🎉 no goals
  -/


lemma egauge_smul_left (hc : c ≠ 0) (s : Set E) (x : E) :
    egauge 𝕜 (c • s) x = egauge 𝕜 s x / ‖c‖₊ := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    hc : Ne c 0
    s : Set E
    x : E
    ⊢ Eq (egauge 𝕜 (HSMul.hSMul c s) x) (HDiv.hDiv (egauge 𝕜 s x) ↑(NNNorm.nnnorm  …
  -/
  refine le_antisymm ?_ (le_egauge_smul_left _ _ _)
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    hc : Ne c 0
    s : Set E
    x : E
    ⊢ LE.le (egauge 𝕜 (HSMul.hSMul c s) x) (HDiv.hDiv (egauge 𝕜 s x) ↑(NNNorm.nnno …
  -/
  rw [ENNReal.le_div_iff_mul_le (by simp [*]) (by simp)]
  calc
    egauge 𝕜 (c • s) x * ‖c‖₊ = egauge 𝕜 (c • s) x / ‖c⁻¹‖₊ := by
      rw [nnnorm_inv, ENNReal.coe_inv (by simpa), div_eq_mul_inv, inv_inv]
    _ ≤ egauge 𝕜 (c⁻¹ • c • s) x := le_egauge_smul_left _ _ _
    _ = egauge 𝕜 s x := by rw [inv_smul_smul₀ hc]


lemma le_egauge_smul_right (c : 𝕜) (s : Set E) (x : E) :
    ‖c‖₊ * egauge 𝕜 s x ≤ egauge 𝕜 s (c • x) := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (egauge 𝕜 s x)) (egauge 𝕜 s (HSMul.hSM …
  -/
  rw [le_egauge_iff]
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    ⊢ ∀ (c_1 : 𝕜), Membership.mem (HSMul.hSMul c_1 s) (HSMul.hSMul c x) → LE.le (H …
  -/
  rintro a ⟨y, hy, hxy⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    x : E
    a : 𝕜
    y : E
    hy : Membership.mem s y
    hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul c x)
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (egauge 𝕜 s x)) ↑(NNNorm.nnnorm a)
  -/
  rcases eq_or_ne c 0 with rfl | hc
    /-
      case intro.intro.inl
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      a : 𝕜
      y : E
      hy : Membership.mem s y
      hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul 0 x)
      ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm 0)) (egauge 𝕜 s x)) ↑(NNNorm.nnnorm a)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      x : E
      a : 𝕜
      y : E
      hy : Membership.mem s y
      hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul c x)
      hc : Ne c 0
      ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (egauge 𝕜 s x)) ↑(NNNorm.nnnorm a)
    -/
  · refine ENNReal.mul_le_of_le_div' <| le_trans ?_ ENNReal.coe_div_le
    /-
      case intro.intro.inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      x : E
      a : 𝕜
      y : E
      hy : Membership.mem s y
      hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul c x)
      hc : Ne c 0
      ⊢ LE.le (egauge 𝕜 s x) ↑(HDiv.hDiv (NNNorm.nnnorm a) (NNNorm.nnnorm c))
    -/
    rw [div_eq_inv_mul, ← nnnorm_inv, ← nnnorm_mul]
    /-
      case intro.intro.inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      x : E
      a : 𝕜
      y : E
      hy : Membership.mem s y
      hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul c x)
      hc : Ne c 0
      ⊢ LE.le (egauge 𝕜 s x) ↑(NNNorm.nnnorm (HMul.hMul (Inv.inv c) a))
    -/
    refine egauge_le_of_mem_smul ⟨y, hy, ?_⟩
    /-
      case intro.intro.inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      x : E
      a : 𝕜
      y : E
      hy : Membership.mem s y
      hxy : Eq ((fun x => HSMul.hSMul a x) y) (HSMul.hSMul c x)
      hc : Ne c 0
      ⊢ Eq ((fun x => HSMul.hSMul (HMul.hMul (Inv.inv c) a) x) y) x
    -/
    simp only [mul_smul, hxy, inv_smul_smul₀ hc]
    /-
      🎉 no goals
    -/


lemma egauge_smul_right (h : c = 0 → s.Nonempty) (x : E) :
    egauge 𝕜 s (c • x) = ‖c‖₊ * egauge 𝕜 s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    h : Eq c 0 → s.Nonempty
    x : E
    ⊢ Eq (egauge 𝕜 s (HSMul.hSMul c x)) (HMul.hMul (↑(NNNorm.nnnorm c)) (egauge 𝕜  …
  -/
  refine le_antisymm ?_ (le_egauge_smul_right c s x)
  /-
    𝕜 : Type u_1
    inst✝² : NormedDivisionRing 𝕜
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    c : 𝕜
    s : Set E
    h : Eq c 0 → s.Nonempty
    x : E
    ⊢ LE.le (egauge 𝕜 s (HSMul.hSMul c x)) (HMul.hMul (↑(NNNorm.nnnorm c)) (egauge …
  -/
  rcases eq_or_ne c 0 with rfl | hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      h : Eq 0 0 → s.Nonempty
      ⊢ LE.le (egauge 𝕜 s (HSMul.hSMul 0 x)) (HMul.hMul (↑(NNNorm.nnnorm 0)) (egauge …
    -/
  · simp [egauge_zero_right _ (h rfl)]
    /-
      🎉 no goals
    -/
  · rw [mul_comm, ← ENNReal.div_le_iff_le_mul (.inl <| by simpa) (.inl ENNReal.coe_ne_top),
      ENNReal.div_eq_inv_mul, ← ENNReal.coe_inv (by simpa), ← nnnorm_inv]
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      h : Eq c 0 → s.Nonempty
      x : E
      hc : Ne c 0
      ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm (Inv.inv c))) (egauge 𝕜 s (HSMul.hSMul c x …
    -/
    refine (le_egauge_smul_right _ _ _).trans_eq ?_
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NormedDivisionRing 𝕜
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      c : 𝕜
      s : Set E
      h : Eq c 0 → s.Nonempty
      x : E
      hc : Ne c 0
      ⊢ Eq (egauge 𝕜 s (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c x))) (egauge 𝕜 s x)
    -/
    rw [inv_smul_smul₀ hc]
    /-
      🎉 no goals
    -/


lemma div_le_egauge_closedBall (r : ℝ≥0) (x : E) : ‖x‖₊ / r ≤ egauge 𝕜 (closedBall 0 r) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    x : E
    ⊢ LE.le (HDiv.hDiv ↑(NNNorm.nnnorm x) ↑r) (egauge 𝕜 (Metric.closedBall 0 ↑r) x)
  -/
  rw [le_egauge_iff]
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    x : E
    ⊢ ∀ (c : 𝕜), Membership.mem (HSMul.hSMul c (Metric.closedBall 0 ↑r)) x → LE.le …
  -/
  rintro c ⟨y, hy, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    c : 𝕜
    y : E
    hy : Membership.mem (Metric.closedBall 0 ↑r) y
    ⊢ LE.le (HDiv.hDiv ↑(NNNorm.nnnorm ((fun x => HSMul.hSMul c x) y)) ↑r) ↑(NNNor …
  -/
  rw [mem_closedBall_zero_iff, ← coe_nnnorm, NNReal.coe_le_coe] at hy
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    c : 𝕜
    y : E
    hy : LE.le (NNNorm.nnnorm y) r
    ⊢ LE.le (HDiv.hDiv ↑(NNNorm.nnnorm ((fun x => HSMul.hSMul c x) y)) ↑r) ↑(NNNor …
  -/
  simp only [nnnorm_smul, ENNReal.coe_mul]
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    c : 𝕜
    y : E
    hy : LE.le (NNNorm.nnnorm y) r
    ⊢ LE.le (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm c) ↑(NNNorm.nnnorm y)) ↑r) ↑(NNN …
  -/
  apply ENNReal.div_le_of_le_mul
  /-
    case intro.intro.h
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : NNReal
    c : 𝕜
    y : E
    hy : LE.le (NNNorm.nnnorm y) r
    ⊢ LE.le (HMul.hMul ↑(NNNorm.nnnorm c) ↑(NNNorm.nnnorm y)) (HMul.hMul ↑(NNNorm. …
  -/
  gcongr
  /-
    🎉 no goals
  -/


lemma le_egauge_closedBall_one (x : E) : ‖x‖₊ ≤ egauge 𝕜 (closedBall 0 1) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ LE.le (↑(NNNorm.nnnorm x)) (egauge 𝕜 (Metric.closedBall 0 1) x)
  -/
  simpa using div_le_egauge_closedBall 𝕜 1 x
  /-
    🎉 no goals
  -/


lemma div_le_egauge_ball (r : ℝ≥0) (x : E) : ‖x‖₊ / r ≤ egauge 𝕜 (ball 0 r) x :=
  (div_le_egauge_closedBall 𝕜 r x).trans <| egauge_anti _ ball_subset_closedBall _


lemma le_egauge_ball_one (x : E) : ‖x‖₊ ≤ egauge 𝕜 (ball 0 1) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ LE.le (↑(NNNorm.nnnorm x)) (egauge 𝕜 (Metric.ball 0 1) x)
  -/
  simpa using div_le_egauge_ball 𝕜 1 x
  /-
    🎉 no goals
  -/


lemma egauge_ball_le_of_one_lt_norm (hc : 1 < ‖c‖) (h₀ : r ≠ 0 ∨ ‖x‖ ≠ 0) :
    egauge 𝕜 (ball 0 r) x ≤ ‖c‖₊ * ‖x‖₊ / r := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    x : E
    r : NNReal
    hc : LT.lt 1 (Norm.norm c)
    h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
    ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
  -/
  letI : NontriviallyNormedField 𝕜 := ⟨c, hc⟩
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    x : E
    r : NNReal
    hc : LT.lt 1 (Norm.norm c)
    h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
    this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
    ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
  -/
  rcases (zero_le r).eq_or_lt with rfl | hr
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NormedField 𝕜
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      c : 𝕜
      x : E
      hc : LT.lt 1 (Norm.norm c)
      this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
      h₀ : Or (Ne 0 0) (Ne (Norm.norm x) 0)
      ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑0) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
    -/
  · rw [ENNReal.coe_zero, ENNReal.div_zero (mul_ne_zero _ _)]
      /-
        case inl
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        hc : LT.lt 1 (Norm.norm c)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        h₀ : Or (Ne 0 0) (Ne (Norm.norm x) 0)
        ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑0) x) Top.top
      -/
    · apply le_top
      /-
        🎉 no goals
      -/
      /-
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        hc : LT.lt 1 (Norm.norm c)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        h₀ : Or (Ne 0 0) (Ne (Norm.norm x) 0)
        ⊢ Ne (↑(NNNorm.nnnorm c)) 0
      -/
    · simpa using one_pos.trans hc
      /-
        🎉 no goals
      -/
      /-
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        hc : LT.lt 1 (Norm.norm c)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        h₀ : Or (Ne 0 0) (Ne (Norm.norm x) 0)
        ⊢ Ne (↑(NNNorm.nnnorm x)) 0
      -/
    · simpa [← NNReal.coe_eq_zero] using h₀
      /-
        🎉 no goals
      -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NormedField 𝕜
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      c : 𝕜
      x : E
      r : NNReal
      hc : LT.lt 1 (Norm.norm c)
      h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
      this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
      hr : LT.lt 0 r
      ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
    -/
  · rcases eq_or_ne ‖x‖ 0 with hx | hx
      /-
        case inr.inl
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        r : NNReal
        hc : LT.lt 1 (Norm.norm c)
        h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        hr : LT.lt 0 r
        hx : Eq (Norm.norm x) 0
        ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
      -/
    · have hx' : ‖x‖₊ = 0 := by rwa [← coe_nnnorm, NNReal.coe_eq_zero] at hx
      /-
        case inr.inl
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        r : NNReal
        hc : LT.lt 1 (Norm.norm c)
        h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        hr : LT.lt 0 r
        hx : Eq (Norm.norm x) 0
        hx' : Eq (NNNorm.nnnorm x) 0
        ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
      -/
      simp [egauge_eq_zero_iff, hx']
      /-
        case inr.inl
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        r : NNReal
        hc : LT.lt 1 (Norm.norm c)
        h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        hr : LT.lt 0 r
        hx : Eq (Norm.norm x) 0
        hx' : Eq (NNNorm.nnnorm x) 0
        ⊢ Filter.Frequently (fun c => Membership.mem (HSMul.hSMul c (Metric.ball 0 ↑r) …
      -/
      refine (frequently_iff_neBot.2 (inferInstance : NeBot (𝓝[≠] (0 : 𝕜)))).mono fun c hc ↦ ?_
      /-
        case inr.inl
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c✝ : 𝕜
        x : E
        r : NNReal
        hc✝ : LT.lt 1 (Norm.norm c✝)
        h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        hr : LT.lt 0 r
        hx : Eq (Norm.norm x) 0
        hx' : Eq (NNNorm.nnnorm x) 0
        c : 𝕜
        hc : Not (Membership.mem (Singleton.singleton 0) c)
        ⊢ Membership.mem (HSMul.hSMul c (Metric.ball 0 ↑r)) x
      -/
      simp [mem_smul_set_iff_inv_smul_mem₀ hc, norm_smul, hx, hr]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        𝕜 : Type u_1
        inst✝² : NormedField 𝕜
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        x : E
        r : NNReal
        hc : LT.lt 1 (Norm.norm c)
        h₀ : Or (Ne r 0) (Ne (Norm.norm x) 0)
        this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.mk ⋯
        hr : LT.lt 0 r
        hx : Ne (Norm.norm x) 0
        ⊢ LE.le (egauge 𝕜 (Metric.ball 0 ↑r) x) (HDiv.hDiv (HMul.hMul ↑(NNNorm.nnnorm  …
      -/
    · rcases rescale_to_shell_semi_normed hc hr hx with ⟨a, ha₀, har, -, hainv⟩
      calc
        egauge 𝕜 (ball 0 r) x ≤ ↑(‖a‖₊⁻¹) :=
          egauge_le_of_smul_mem_of_ne (mem_ball_zero_iff.2 har) ha₀
        _ ≤ ↑(‖c‖₊ * ‖x‖₊ / r) := by rwa [ENNReal.coe_le_coe, div_eq_inv_mul, ← mul_assoc]
        _ ≤ ‖c‖₊ * ‖x‖₊ / r := ENNReal.coe_div_le.trans <| by rw [ENNReal.coe_mul]


lemma egauge_ball_one_le_of_one_lt_norm (hc : 1 < ‖c‖) (x : E) :
    egauge 𝕜 (ball 0 1) x ≤ ‖c‖₊ * ‖x‖₊ := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    x : E
    ⊢ LE.le (egauge 𝕜 (Metric.ball 0 1) x) (HMul.hMul ↑(NNNorm.nnnorm c) ↑(NNNorm. …
  -/
  simpa using egauge_ball_le_of_one_lt_norm hc (.inl one_ne_zero)
  /-
    🎉 no goals
  -/



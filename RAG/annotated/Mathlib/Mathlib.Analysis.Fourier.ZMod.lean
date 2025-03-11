/--
The discrete Fourier transform on `ℤ / N ℤ` (with the counting measure). This definition is
private because it is superseded by the bundled `LinearEquiv` version.
-/
private noncomputable def auxDFT (Φ : ZMod N → E) (k : ZMod N) : E :=
  ∑ j : ZMod N, stdAddChar (-(j * k)) • Φ j


private lemma auxDFT_neg (Φ : ZMod N → E) : auxDFT (fun j ↦ Φ (-j)) = fun k ↦ auxDFT Φ (-k) := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    ⊢ Eq (ZMod.auxDFT fun j => Φ (Neg.neg j)) fun k => ZMod.auxDFT Φ (Neg.neg k)
  -/
  ext1 k; simpa only [auxDFT] using
    Fintype.sum_equiv (Equiv.neg _) _ _ (fun j ↦ by rw [Equiv.neg_apply, neg_mul_neg])


/-- Fourier inversion formula, discrete case. -/
private lemma auxDFT_auxDFT (Φ : ZMod N → E) : auxDFT (auxDFT Φ) = fun j ↦ (N : ℂ) • Φ (-j) := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    ⊢ Eq (ZMod.auxDFT (ZMod.auxDFT Φ)) fun j => HSMul.hSMul (↑N) (Φ (Neg.neg j))
  -/
  ext1 j
  simp only [auxDFT, mul_comm _ j, smul_sum, ← smul_assoc, smul_eq_mul, ← map_add_eq_mul, ←
    neg_add, ← add_mul]
  /-
    case h
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    j : ZMod N
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HSMul.hSMul (ZMod.st …
  -/
  rw [sum_comm]
  /-
    case h
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    j : ZMod N
    ⊢ Eq (Finset.univ.sum fun y => Finset.univ.sum fun x => HSMul.hSMul (ZMod.stdA …
  -/
  simp only [← sum_smul, ← neg_mul]
  have h1 (t : ZMod N) : ∑ i, stdAddChar (t * i) = if t = 0 then ↑N else 0 := by
    split_ifs with h
    · simp only [h, zero_mul, map_zero_eq_one, sum_const, card_univ, card,
        nsmul_eq_mul, mul_one]
    · exact sum_eq_zero_of_ne_one (isPrimitive_stdAddChar N h)
  have h2 (x j : ZMod N) : -(j + x) = 0 ↔ x = -j := by
    rw [neg_add, add_comm, add_eq_zero_iff_neg_eq, neg_neg]
  /-
    case h
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    j : ZMod N
    h1 : ∀ (t : ZMod N), Eq (Finset.univ.sum fun i => ZMod.stdAddChar (HMul.hMul t …
    h2 : ∀ (x j : ZMod N), Iff (Eq (Neg.neg (HAdd.hAdd j x)) 0) (Eq x (Neg.neg j))
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (Finset.univ.sum fun i => ZMod.stdA …
  -/
  simp only [h1, h2, ite_smul, zero_smul, sum_ite_eq', mem_univ, ite_true]
  /-
    🎉 no goals
  -/


private lemma auxDFT_smul (c : ℂ) (Φ : ZMod N → E) :
    auxDFT (c • Φ) = c • auxDFT Φ := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    c : Complex
    Φ : ZMod N → E
    ⊢ Eq (ZMod.auxDFT (HSMul.hSMul c Φ)) (HSMul.hSMul c (ZMod.auxDFT Φ))
  -/
  ext; simp only [Pi.smul_def, auxDFT, smul_sum, smul_comm c]
       /-
         🎉 no goals
       -/


/--
The discrete Fourier transform on `ℤ / N ℤ` (with the counting measure), bundled as a linear
equivalence.
-/
noncomputable def dft : (ZMod N → E) ≃ₗ[ℂ] (ZMod N → E) where
  toFun := auxDFT
  map_add' Φ₁ Φ₂ := by
    /-
      N : Nat
      inst✝² : NeZero N
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      Φ₁ Φ₂ : ZMod N → E
      ⊢ Eq (ZMod.auxDFT (HAdd.hAdd Φ₁ Φ₂)) (HAdd.hAdd (ZMod.auxDFT Φ₁) (ZMod.auxDFT  …
    -/
    ext; simp only [auxDFT, Pi.add_def, smul_add, sum_add_distrib]
         /-
           🎉 no goals
         -/
  map_smul' c Φ := by
    /-
      N : Nat
      inst✝² : NeZero N
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      c : Complex
      Φ : ZMod N → E
      ⊢ Eq ({ toFun := ZMod.auxDFT, map_add' := ⋯ }.toFun (HSMul.hSMul c Φ)) (HSMul. …
    -/
    ext; simp only [auxDFT, Pi.smul_apply, RingHom.id_apply, smul_sum, smul_comm c]
         /-
           🎉 no goals
         -/
  invFun Φ k := (N : ℂ)⁻¹ • auxDFT Φ (-k)
  left_inv Φ := by
    /-
      N : Nat
      inst✝² : NeZero N
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      Φ : ZMod N → E
      ⊢ Eq ((fun Φ k => HSMul.hSMul (Inv.inv ↑N) (ZMod.auxDFT Φ (Neg.neg k))) ({ toF …
    -/
    simp only [auxDFT_auxDFT, neg_neg, ← mul_smul, inv_mul_cancel₀ (NeZero.ne _), one_smul]
    /-
      🎉 no goals
    -/
  right_inv Φ := by
    /-
      N : Nat
      inst✝² : NeZero N
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      Φ : ZMod N → E
      ⊢ Eq ({ toFun := ZMod.auxDFT, map_add' := ⋯, map_smul' := ⋯ }.toFun ((fun Φ k  …
    -/
    ext1 j
    simp only [← Pi.smul_def, auxDFT_smul, auxDFT_neg, auxDFT_auxDFT, neg_neg, ← mul_smul,
      inv_mul_cancel₀ (NeZero.ne _), one_smul]


@[inherit_doc] scoped notation "𝓕" => dft


/-- The inverse Fourier transform on `ZMod N`. -/
scoped notation "𝓕⁻" => LinearEquiv.symm dft


lemma dft_apply (Φ : ZMod N → E) (k : ZMod N) :
    𝓕 Φ k = ∑ j : ZMod N, stdAddChar (-(j * k)) • Φ j :=
  rfl


lemma dft_def (Φ : ZMod N → E) :
    𝓕 Φ = fun k ↦ ∑ j : ZMod N, stdAddChar (-(j * k)) • Φ j :=
  rfl


lemma invDFT_apply (Ψ : ZMod N → E) (k : ZMod N) :
    𝓕⁻ Ψ k = (N : ℂ)⁻¹ • ∑ j : ZMod N, stdAddChar (j * k) • Ψ j := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Ψ : ZMod N → E
    k : ZMod N
    ⊢ Eq (ZMod.dft.symm Ψ k) (HSMul.hSMul (Inv.inv ↑N) (Finset.univ.sum fun j => H …
  -/
  simp only [dft, LinearEquiv.coe_symm_mk, auxDFT, mul_neg, neg_neg]
  /-
    🎉 no goals
  -/


lemma invDFT_def (Ψ : ZMod N → E) :
    𝓕⁻ Ψ = fun k ↦ (N : ℂ)⁻¹ • ∑ j : ZMod N, stdAddChar (j * k) • Ψ j :=
  funext <| invDFT_apply Ψ


lemma invDFT_apply' (Ψ : ZMod N → E) (k : ZMod N) : 𝓕⁻ Ψ k = (N : ℂ)⁻¹ • 𝓕 Ψ (-k) :=
  rfl


lemma invDFT_def' (Ψ : ZMod N → E) : 𝓕⁻ Ψ = fun k ↦ (N : ℂ)⁻¹ • 𝓕 Ψ (-k) :=
  rfl


lemma dft_apply_zero (Φ : ZMod N → E) : 𝓕 Φ 0 = ∑ j, Φ j := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    ⊢ Eq (ZMod.dft Φ 0) (Finset.univ.sum fun j => Φ j)
  -/
  simp only [dft_apply, mul_zero, neg_zero, map_zero_eq_one, one_smul]
  /-
    🎉 no goals
  -/


/--
The discrete Fourier transform agrees with the general one (assuming the target space is a complete
normed space).
-/
lemma dft_eq_fourier {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E] [CompleteSpace E]
    (Φ : ZMod N → E) (k : ZMod N) :
    𝓕 Φ k = Fourier.fourierIntegral toCircle Measure.count Φ k := by
  simp only [dft_apply, stdAddChar_apply, Fourier.fourierIntegral_def, Circle.smul_def,
    integral_countable' <| .of_finite .., Measure.count_singleton, ENNReal.one_toReal, one_smul,
    tsum_fintype]


lemma dft_const_smul {R : Type*} [DistribSMul R E] [SMulCommClass R ℂ E] (r : R) (Φ : ZMod N → E) :
    𝓕 (r • Φ) = r • 𝓕 Φ := by
  /-
    N : Nat
    inst✝⁴ : NeZero N
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Complex E
    R : Type u_2
    inst✝¹ : DistribSMul R E
    inst✝ : SMulCommClass R Complex E
    r : R
    Φ : ZMod N → E
    ⊢ Eq (ZMod.dft (HSMul.hSMul r Φ)) (HSMul.hSMul r (ZMod.dft Φ))
  -/
  simp only [Pi.smul_def, dft_def, smul_sum, smul_comm]
  /-
    🎉 no goals
  -/


lemma dft_smul_const {R : Type*} [Ring R] [Module ℂ R] [Module R E] [IsScalarTower ℂ R E]
    (Φ : ZMod N → R) (e : E) :
    𝓕 (fun j ↦ Φ j • e) = fun k ↦ 𝓕 Φ k • e := by
  /-
    N : Nat
    inst✝⁶ : NeZero N
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Complex E
    R : Type u_2
    inst✝³ : Ring R
    inst✝² : Module Complex R
    inst✝¹ : Module R E
    inst✝ : IsScalarTower Complex R E
    Φ : ZMod N → R
    e : E
    ⊢ Eq (ZMod.dft fun j => HSMul.hSMul (Φ j) e) fun k => HSMul.hSMul (ZMod.dft Φ  …
  -/
  simp only [dft_def, sum_smul, smul_assoc]
  /-
    🎉 no goals
  -/


lemma dft_const_mul {R : Type*} [Ring R] [Algebra ℂ R] (r : R) (Φ : ZMod N → R) :
    𝓕 (fun j ↦ r * Φ j) = fun k ↦ r * 𝓕 Φ k :=
  dft_const_smul r Φ


lemma dft_mul_const {R : Type*} [Ring R] [Algebra ℂ R] (Φ : ZMod N → R) (r : R) :
    𝓕 (fun j ↦ Φ j * r) = fun k ↦ 𝓕 Φ k * r :=
  dft_smul_const Φ r


lemma dft_comp_neg (Φ : ZMod N → E) : 𝓕 (fun j ↦ Φ (-j)) = fun k ↦ 𝓕 Φ (-k) :=
  auxDFT_neg ..


/-- Fourier inversion formula, discrete case. -/
lemma dft_dft (Φ : ZMod N → E) : 𝓕 (𝓕 Φ) = fun j ↦ (N : ℂ) • Φ (-j) :=
  auxDFT_auxDFT ..


lemma dft_comp_unitMul (Φ : ZMod N → E) (u : (ZMod N)ˣ) (k : ZMod N) :
    𝓕 (fun j ↦ Φ (u.val * j)) k = 𝓕 Φ (u⁻¹.val * k) := by
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    u : Units (ZMod N)
    k : ZMod N
    ⊢ Eq (ZMod.dft (fun j => Φ (HMul.hMul (↑u) j)) k) (ZMod.dft Φ (HMul.hMul (↑(In …
  -/
  refine Fintype.sum_equiv u.mulLeft _ _ fun x ↦ ?_
  /-
    N : Nat
    inst✝² : NeZero N
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    Φ : ZMod N → E
    u : Units (ZMod N)
    k x : ZMod N
    ⊢ Eq (HSMul.hSMul (ZMod.stdAddChar (Neg.neg (HMul.hMul x k))) ((fun j => Φ (HM …
  -/
  simp only [mul_comm u.val, u.mulLeft_apply, ← mul_assoc, u.mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


/-- The discrete Fourier transform of `Φ` is even if and only if `Φ` itself is. -/
lemma dft_even_iff {Φ : ZMod N → ℂ} : (𝓕 Φ).Even ↔ Φ.Even := by
  have h {f : ZMod N → ℂ} (hf : f.Even) : (𝓕 f).Even := by
    simp only [Function.Even, ← congr_fun (dft_comp_neg f), funext hf, implies_true]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    h : ∀ {f : ZMod N → Complex}, Function.Even f → Function.Even (ZMod.dft f)
    ⊢ Iff (Function.Even (ZMod.dft Φ)) (Function.Even Φ)
  -/
  refine ⟨fun hΦ x ↦ ?_, h⟩
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    h : ∀ {f : ZMod N → Complex}, Function.Even f → Function.Even (ZMod.dft f)
    hΦ : Function.Even (ZMod.dft Φ)
    x : ZMod N
    ⊢ Eq (Φ (Neg.neg x)) (Φ x)
  -/
  simpa only [neg_neg, smul_right_inj (NeZero.ne (N : ℂ)), dft_dft] using h hΦ (-x)
  /-
    🎉 no goals
  -/


/-- The discrete Fourier transform of `Φ` is odd if and only if `Φ` itself is. -/
lemma dft_odd_iff {Φ : ZMod N → ℂ} : (𝓕 Φ).Odd ↔ Φ.Odd := by
  have h {f : ZMod N → ℂ} (hf : f.Odd) : (𝓕 f).Odd := by
    simp only [Function.Odd, ← congr_fun (dft_comp_neg f), funext hf, ← Pi.neg_apply, map_neg,
      implies_true]
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    h : ∀ {f : ZMod N → Complex}, Function.Odd f → Function.Odd (ZMod.dft f)
    ⊢ Iff (Function.Odd (ZMod.dft Φ)) (Function.Odd Φ)
  -/
  refine ⟨fun hΦ x ↦ ?_, h⟩
  /-
    N : Nat
    inst✝ : NeZero N
    Φ : ZMod N → Complex
    h : ∀ {f : ZMod N → Complex}, Function.Odd f → Function.Odd (ZMod.dft f)
    hΦ : Function.Odd (ZMod.dft Φ)
    x : ZMod N
    ⊢ Eq (Φ (Neg.neg x)) (Neg.neg (Φ x))
  -/
  simpa only [neg_neg, dft_dft, ← smul_neg, smul_right_inj (NeZero.ne (N : ℂ))] using h hΦ (-x)
  /-
    🎉 no goals
  -/


lemma fourierTransform_eq_gaussSum_mulShift (χ : DirichletCharacter ℂ N) (k : ZMod N) :
    𝓕 χ k = gaussSum χ (stdAddChar.mulShift (-k)) := by
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    k : ZMod N
    ⊢ Eq (ZMod.dft (⇑χ) k) (gaussSum χ (ZMod.stdAddChar.mulShift (Neg.neg k)))
  -/
  simp only [dft_apply, smul_eq_mul]
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    k : ZMod N
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (ZMod.stdAddChar (Neg.neg (HMul.hMul  …
  -/
  congr 1 with j
  /-
    case e_f.h
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    k j : ZMod N
    ⊢ Eq (HMul.hMul (ZMod.stdAddChar (Neg.neg (HMul.hMul j k))) (χ j)) (HMul.hMul  …
  -/
  rw [mulShift_apply, mul_comm j, neg_mul, stdAddChar_apply, mul_comm (χ _)]
  /-
    🎉 no goals
  -/


/-- For a primitive Dirichlet character `χ`, the Fourier transform of `χ` is a constant multiple
of `χ⁻¹` (and the constant is essentially the Gauss sum). -/
lemma IsPrimitive.fourierTransform_eq_inv_mul_gaussSum {χ : DirichletCharacter ℂ N}
    (hχ : IsPrimitive χ) (k : ZMod N) :
    𝓕 χ k = χ⁻¹ (-k) * gaussSum χ stdAddChar := by
  /-
    N : Nat
    inst✝ : NeZero N
    χ : DirichletCharacter Complex N
    hχ : χ.IsPrimitive
    k : ZMod N
    ⊢ Eq (ZMod.dft (⇑χ) k) (HMul.hMul ((Inv.inv χ) (Neg.neg k)) (gaussSum χ ZMod.s …
  -/
  rw [fourierTransform_eq_gaussSum_mulShift, gaussSum_mulShift_of_isPrimitive _ hχ]
  /-
    🎉 no goals
  -/



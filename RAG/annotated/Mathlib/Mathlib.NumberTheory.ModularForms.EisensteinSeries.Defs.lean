/-- The set of pairs of coprime integers congruent to `a` mod `N`. -/
def gammaSet := {v : Fin 2 → ℤ | (↑) ∘ v = a ∧ IsCoprime (v 0) (v 1)}


lemma pairwise_disjoint_gammaSet : Pairwise (Disjoint on gammaSet N) := by
  /-
    N : Nat
    ⊢ Pairwise (Function.onFun Disjoint (EisensteinSeries.gammaSet N))
  -/
  refine fun u v huv ↦ ?_
  /-
    N : Nat
    u v : Fin 2 → ZMod N
    huv : Ne u v
    ⊢ Function.onFun Disjoint (EisensteinSeries.gammaSet N) u v
  -/
  contrapose! huv
  /-
    N : Nat
    u v : Fin 2 → ZMod N
    huv : Not (Function.onFun Disjoint (EisensteinSeries.gammaSet N) u v)
    ⊢ Eq u v
  -/
  obtain ⟨f, hf⟩ := Set.not_disjoint_iff.mp huv
  /-
    case intro
    N : Nat
    u v : Fin 2 → ZMod N
    huv : Not (Function.onFun Disjoint (EisensteinSeries.gammaSet N) u v)
    f : Fin 2 → Int
    hf : And (Membership.mem (EisensteinSeries.gammaSet N u) f) (Membership.mem (E …
    ⊢ Eq u v
  -/
  exact hf.1.1.symm.trans hf.2.1
  /-
    🎉 no goals
  -/


/-- For level `N = 1`, the gamma sets are all equal. -/
lemma gammaSet_one_eq (a a' : Fin 2 → ZMod 1) : gammaSet 1 a = gammaSet 1 a' :=
  congr_arg _ (Subsingleton.elim _ _)


/-- For level `N = 1`, the gamma sets are all equivalent; this is the equivalence. -/
def gammaSet_one_equiv (a a' : Fin 2 → ZMod 1) : gammaSet 1 a ≃ gammaSet 1 a' :=
  Equiv.Set.ofEq (gammaSet_one_eq a a')


/-- Right-multiplying by `γ ∈ SL(2, ℤ)` sends `gammaSet N a` to `gammaSet N (a ᵥ* γ)`. -/
lemma vecMul_SL2_mem_gammaSet {v : Fin 2 → ℤ} (hv : v ∈ gammaSet N a) (γ : SL(2, ℤ)) :
    v ᵥ* γ ∈ gammaSet N (a ᵥ* γ) := by
  /-
    N : Nat
    a : Fin 2 → ZMod N
    v : Fin 2 → Int
    hv : Membership.mem (EisensteinSeries.gammaSet N a) v
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Membership.mem (EisensteinSeries.gammaSet N (Matrix.vecMul a ↑((Matrix.Speci …
  -/
  refine ⟨?_, hv.2.vecMulSL γ⟩
  /-
    N : Nat
    a : Fin 2 → ZMod N
    v : Fin 2 → Int
    hv : Membership.mem (EisensteinSeries.gammaSet N a) v
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (Function.comp Int.cast (Matrix.vecMul v ↑γ)) (Matrix.vecMul a ↑((Matrix. …
  -/
  have := RingHom.map_vecMul (m := Fin 2) (n := Fin 2) (Int.castRingHom (ZMod N)) γ v
  /-
    N : Nat
    a : Fin 2 → ZMod N
    v : Fin 2 → Int
    hv : Membership.mem (EisensteinSeries.gammaSet N a) v
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    this : ∀ (i : Fin 2), Eq ((Int.castRingHom (ZMod N)) (Matrix.vecMul v (↑γ) i)) …
    ⊢ Eq (Function.comp Int.cast (Matrix.vecMul v ↑γ)) (Matrix.vecMul a ↑((Matrix. …
  -/
  simp only [eq_intCast, Int.coe_castRingHom] at this
  /-
    N : Nat
    a : Fin 2 → ZMod N
    v : Fin 2 → Int
    hv : Membership.mem (EisensteinSeries.gammaSet N a) v
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    this : ∀ (i : Fin 2), Eq (↑(Matrix.vecMul v (↑γ) i)) (Matrix.vecMul (Function. …
    ⊢ Eq (Function.comp Int.cast (Matrix.vecMul v ↑γ)) (Matrix.vecMul a ↑((Matrix. …
  -/
  simp_rw [Function.comp_def, this, hv.1]
  /-
    N : Nat
    a : Fin 2 → ZMod N
    v : Fin 2 → Int
    hv : Membership.mem (EisensteinSeries.gammaSet N a) v
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    this : ∀ (i : Fin 2), Eq (↑(Matrix.vecMul v (↑γ) i)) (Matrix.vecMul (Function. …
    ⊢ Eq (fun x => Matrix.vecMul a ((↑γ).map fun x => ↑x) x) (Matrix.vecMul a ↑((M …
  -/
  simp
  /-
    🎉 no goals
  -/


variable (a) in
/-- The bijection between `GammaSets` given by multiplying by an element of `SL(2, ℤ)`. -/
def gammaSetEquiv (γ : SL(2, ℤ)) : gammaSet N a ≃ gammaSet N (a ᵥ* γ) where
  toFun v := ⟨v.1 ᵥ* γ, vecMul_SL2_mem_gammaSet v.2 γ⟩
  invFun v := ⟨v.1 ᵥ* ↑(γ⁻¹), by
      /-
        N : Nat
        a : Fin 2 → ZMod N
        γ : Matrix.SpecialLinearGroup (Fin 2) Int
        v : ↑(EisensteinSeries.gammaSet N (Matrix.vecMul a ↑((Matrix.SpecialLinearGrou …
        ⊢ Membership.mem (EisensteinSeries.gammaSet N a) (Matrix.vecMul ↑v ↑(Inv.inv γ))
      -/
      have := vecMul_SL2_mem_gammaSet v.2 γ⁻¹
      /-
        N : Nat
        a : Fin 2 → ZMod N
        γ : Matrix.SpecialLinearGroup (Fin 2) Int
        v : ↑(EisensteinSeries.gammaSet N (Matrix.vecMul a ↑((Matrix.SpecialLinearGrou …
        this : Membership.mem (EisensteinSeries.gammaSet N (Matrix.vecMul (Matrix.vecM …
        ⊢ Membership.mem (EisensteinSeries.gammaSet N a) (Matrix.vecMul ↑v ↑(Inv.inv γ))
      -/
      rw [vecMul_vecMul, ← SpecialLinearGroup.coe_mul] at this
      simpa only [SpecialLinearGroup.map_apply_coe, RingHom.mapMatrix_apply, Int.coe_castRingHom,
        map_inv, mul_inv_cancel, SpecialLinearGroup.coe_one, vecMul_one]⟩
  left_inv v := by simp_rw [vecMul_vecMul, ← SpecialLinearGroup.coe_mul, mul_inv_cancel,
    SpecialLinearGroup.coe_one, vecMul_one]
  right_inv v := by simp_rw [vecMul_vecMul, ← SpecialLinearGroup.coe_mul, inv_mul_cancel,
    SpecialLinearGroup.coe_one, vecMul_one]


/-- The function on `(Fin 2 → ℤ)` whose sum defines an Eisenstein series. -/
def eisSummand (k : ℤ) (v : Fin 2 → ℤ) (z : ℍ) : ℂ := (v 0 * z.1 + v 1) ^ (-k)


/-- How the `eisSummand` function changes under the Moebius action. -/
theorem eisSummand_SL2_apply (k : ℤ) (i : (Fin 2 → ℤ)) (A : SL(2, ℤ)) (z : ℍ) :
    eisSummand k i (A • z) = (z.denom A) ^ k * eisSummand k (i ᵥ* A) z := by
  /-
    k : Int
    i : Fin 2 → Int
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    ⊢ Eq (EisensteinSeries.eisSummand k i (HSMul.hSMul A z)) (HMul.hMul (HPow.hPow …
  -/
  simp only [eisSummand, vecMul, vec2_dotProduct]
  /-
    k : Int
    i : Fin 2 → Int
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul ↑(i 0) ↑(HSMul.hSMul A z)) ↑(i 1)) (Neg. …
  -/
  push_cast
  have h (a b c d u v : ℂ) (hc : c * z + d ≠ 0) : (u * ((a * z + b) / (c * z + d)) + v) ^ (-k) =
      (c * z + d) ^ k * ((u * a + v * c) * z + (u * b + v * d)) ^ (-k) := by
    field_simp [hc]
    ring_nf
  /-
    k : Int
    i : Fin 2 → Int
    A : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    h : ∀ (a b c d u v : Complex), Ne (HAdd.hAdd (HMul.hMul c ↑z) d) 0 → Eq (HPow. …
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul ↑(i 0) ↑(HSMul.hSMul A z)) ↑(i 1)) (Neg. …
  -/
  apply h (hc := z.denom_ne_zero A)
  /-
    🎉 no goals
  -/


/-- An Eisenstein series of weight `k` and level `Γ(N)`, with congruence condition `a`. -/
def _root_.eisensteinSeries (k : ℤ) (z : ℍ) : ℂ := ∑' x : gammaSet N a, eisSummand k x z


lemma eisensteinSeries_slash_apply (k : ℤ) (γ : SL(2, ℤ)) :
    eisensteinSeries a k ∣[k] γ = eisensteinSeries (a ᵥ* γ) k := by
  /-
    N : Nat
    a : Fin 2 → ZMod N
    k : Int
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (SlashAction.map Complex k γ (eisensteinSeries a k)) (eisensteinSeries (M …
  -/
  ext1 z
  simp_rw [SL_slash, slash_def, slash, ModularGroup.det_coe, ofReal_one, one_zpow, mul_one,
    zpow_neg, mul_inv_eq_iff_eq_mul₀ (zpow_ne_zero _ <| z.denom_ne_zero _), mul_comm,
    eisensteinSeries, ← ModularGroup.sl_moeb, eisSummand_SL2_apply, tsum_mul_left]
  /-
    case h
    N : Nat
    a : Fin 2 → ZMod N
    k : Int
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    ⊢ Eq (HMul.hMul (HPow.hPow (UpperHalfPlane.denom (↑γ) z) k) (tsum fun x => Eis …
  -/
  exact congr_arg (_ * ·) <| (gammaSetEquiv a γ).tsum_eq (eisSummand k · z)
  /-
    🎉 no goals
  -/


/-- The SlashInvariantForm defined by an Eisenstein series of weight `k : ℤ`, level `Γ(N)`,
  and congruence condition given by `a : Fin 2 → ZMod N`. -/
def eisensteinSeries_SIF (k : ℤ) : SlashInvariantForm (Gamma N) k where
  toFun := eisensteinSeries a k
  slash_action_eq' A hA := by simp only [eisensteinSeries_slash_apply, Gamma_mem'.mp hA,
    SpecialLinearGroup.coe_one, vecMul_one]


lemma eisensteinSeries_SIF_apply (k : ℤ) (z : ℍ) :
    eisensteinSeries_SIF a k z = eisensteinSeries a k z := rfl



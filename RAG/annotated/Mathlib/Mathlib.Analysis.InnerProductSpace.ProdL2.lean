noncomputable instance instProdInnerProductSpace :
    InnerProductSpace 𝕜 (WithLp 2 (E × F)) where
  inner x y := inner x.fst y.fst + inner x.snd y.snd
  norm_sq_eq_inner x := by
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x : WithLp 2 (Prod E F)
      ⊢ Eq (HPow.hPow (Norm.norm x) 2) (RCLike.re (Inner.inner x x))
    -/
    simp [prod_norm_sq_eq_of_L2, ← norm_sq_eq_inner]
    /-
      🎉 no goals
    -/
  conj_symm x y := by
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x y : WithLp 2 (Prod E F)
      ⊢ Eq ((starRingEnd 𝕜) (Inner.inner y x)) (Inner.inner x y)
    -/
    simp
    /-
      🎉 no goals
    -/
  add_left x y z := by
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x y z : WithLp 2 (Prod E F)
      ⊢ Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inner x z) (Inner.inner …
    -/
    simp only [add_fst, add_snd, inner_add_left]
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x y z : WithLp 2 (Prod E F)
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Inner.inner x.1 z.1) (Inner.inner y.1 z.1)) (HAdd. …
    -/
    ring
    /-
      🎉 no goals
    -/
  smul_left x y r := by
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x y : WithLp 2 (Prod E F)
      r : 𝕜
      ⊢ Eq (Inner.inner (HSMul.hSMul r x) y) (HMul.hMul ((starRingEnd 𝕜) r) (Inner.i …
    -/
    simp only [smul_fst, inner_smul_left, smul_snd]
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace 𝕜 F
      x y : WithLp 2 (Prod E F)
      r : 𝕜
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((starRingEnd 𝕜) r) (Inner.inner x.1 y.1)) (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/


@[simp]
theorem prod_inner_apply (x y : WithLp 2 (E × F)) :
    inner (𝕜 := 𝕜) x y = inner x.fst y.fst + inner x.snd y.snd := rfl


/-- The product of two orthonormal bases is a basis for the L2-product. -/
def prod (v : OrthonormalBasis ι₁ 𝕜 E) (w : OrthonormalBasis ι₂ 𝕜 F) :
    OrthonormalBasis (ι₁ ⊕ ι₂) 𝕜 (WithLp 2 (E × F)) :=
  ((v.toBasis.prod w.toBasis).map (WithLp.linearEquiv 2 𝕜 (E × F)).symm).toOrthonormalBasis
  (by
    /-
      𝕜 : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : Fintype ι₁
      inst✝ : Fintype ι₂
      v : OrthonormalBasis ι₁ 𝕜 E
      w : OrthonormalBasis ι₂ 𝕜 F
      ⊢ Orthonormal 𝕜 ⇑((v.toBasis.prod w.toBasis).map (WithLp.linearEquiv 2 𝕜 (Prod …
    -/
    constructor
      /-
        case left
        𝕜 : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : InnerProductSpace 𝕜 F
        inst✝¹ : Fintype ι₁
        inst✝ : Fintype ι₂
        v : OrthonormalBasis ι₁ 𝕜 E
        w : OrthonormalBasis ι₂ 𝕜 F
        ⊢ ∀ (i : Sum ι₁ ι₂), Eq (Norm.norm (((v.toBasis.prod w.toBasis).map (WithLp.li …
      -/
    · simp only [Sum.forall, norm_eq_sqrt_inner (𝕜 := 𝕜), Real.sqrt_eq_one]
      /-
        case left
        𝕜 : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : InnerProductSpace 𝕜 F
        inst✝¹ : Fintype ι₁
        inst✝ : Fintype ι₂
        v : OrthonormalBasis ι₁ 𝕜 E
        w : OrthonormalBasis ι₂ 𝕜 F
        ⊢ And (∀ (a : ι₁), Eq (RCLike.re (Inner.inner (((v.toBasis.prod w.toBasis).map …
      -/
      simp [← Real.sqrt_eq_one, ← norm_eq_sqrt_inner (𝕜 := 𝕜), v.orthonormal.1, w.orthonormal.1]
      /-
        🎉 no goals
      -/
      /-
        case right
        𝕜 : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : InnerProductSpace 𝕜 F
        inst✝¹ : Fintype ι₁
        inst✝ : Fintype ι₂
        v : OrthonormalBasis ι₁ 𝕜 E
        w : OrthonormalBasis ι₂ 𝕜 F
        ⊢ Pairwise fun i j => Eq (Inner.inner (((v.toBasis.prod w.toBasis).map (WithLp …
      -/
    · unfold Pairwise
      simp only [ne_eq, Basis.map_apply, Basis.prod_apply, LinearMap.coe_inl,
        OrthonormalBasis.coe_toBasis, LinearMap.coe_inr, WithLp.linearEquiv_symm_apply,
        WithLp.prod_inner_apply, WithLp.equiv_symm_fst, WithLp.equiv_symm_snd, Sum.forall,
        Sum.elim_inl, Function.comp_apply, inner_zero_right, add_zero, Sum.elim_inr, zero_add,
        Sum.inl.injEq, not_false_eq_true, inner_zero_left, forall_true_left, implies_true, and_true,
        Sum.inr.injEq, true_and]
      /-
        case right
        𝕜 : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        E : Type u_4
        F : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : InnerProductSpace 𝕜 F
        inst✝¹ : Fintype ι₁
        inst✝ : Fintype ι₂
        v : OrthonormalBasis ι₁ 𝕜 E
        w : OrthonormalBasis ι₂ 𝕜 F
        ⊢ And (∀ (a a_1 : ι₁), Not (Eq a a_1) → Eq (Inner.inner (v a) (v a_1)) 0) (∀ ( …
      -/
      exact ⟨v.orthonormal.2, w.orthonormal.2⟩)
      /-
        🎉 no goals
      -/


@[simp] theorem prod_apply (v : OrthonormalBasis ι₁ 𝕜 E) (w : OrthonormalBasis ι₂ 𝕜 F) :
    ∀ i : ι₁ ⊕ ι₂, v.prod w i =
      Sum.elim ((LinearMap.inl 𝕜 E F) ∘ v) ((LinearMap.inr 𝕜 E F) ∘ w) i := by
  /-
    𝕜 : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : Fintype ι₁
    inst✝ : Fintype ι₂
    v : OrthonormalBasis ι₁ 𝕜 E
    w : OrthonormalBasis ι₂ 𝕜 F
    ⊢ ∀ (i : Sum ι₁ ι₂), Eq ((v.prod w) i) (Sum.elim (Function.comp ⇑(LinearMap.in …
  -/
  rw [Sum.forall]
  /-
    𝕜 : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : Fintype ι₁
    inst✝ : Fintype ι₂
    v : OrthonormalBasis ι₁ 𝕜 E
    w : OrthonormalBasis ι₂ 𝕜 F
    ⊢ And (∀ (a : ι₁), Eq ((v.prod w) (Sum.inl a)) (Sum.elim (Function.comp ⇑(Line …
  -/
  unfold OrthonormalBasis.prod
  /-
    𝕜 : Type u_1
    ι₁ : Type u_2
    ι₂ : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : Fintype ι₁
    inst✝ : Fintype ι₂
    v : OrthonormalBasis ι₁ 𝕜 E
    w : OrthonormalBasis ι₂ 𝕜 F
    ⊢ And (∀ (a : ι₁), Eq ((((v.toBasis.prod w.toBasis).map (WithLp.linearEquiv 2  …
  -/
  aesop
  /-
    🎉 no goals
  -/



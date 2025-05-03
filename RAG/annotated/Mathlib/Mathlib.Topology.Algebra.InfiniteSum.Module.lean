theorem HasSum.const_smul {a : α} (b : γ) (hf : HasSum f a) : HasSum (fun i ↦ b • f i) (b • a) :=
  hf.map (DistribMulAction.toAddMonoidHom α _) <| continuous_const_smul _


theorem Summable.const_smul (b : γ) (hf : Summable f) : Summable fun i ↦ b • f i :=
  (hf.hasSum.const_smul _).summable


/-- Infinite sums commute with scalar multiplication. Version for scalars living in a `Monoid`, but
  requiring a summability hypothesis. -/
theorem tsum_const_smul [T2Space α] (b : γ) (hf : Summable f) : ∑' i, b • f i = b • ∑' i, f i :=
  (hf.hasSum.const_smul _).tsum_eq


/-- Infinite sums commute with scalar multiplication. Version for scalars living in a `Group`, but
  not requiring any summability hypothesis. -/
lemma tsum_const_smul' {γ : Type*} [Group γ] [DistribMulAction γ α] [ContinuousConstSMul γ α]
    [T2Space α] (g : γ) : ∑' (i : β), g • f i = g • ∑' (i : β), f i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g (tsum fun i => f i))
  -/
  by_cases hf : Summable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : AddCommMonoid α
      f : β → α
      γ : Type u_5
      inst✝³ : Group γ
      inst✝² : DistribMulAction γ α
      inst✝¹ : ContinuousConstSMul γ α
      inst✝ : T2Space α
      g : γ
      hf : Summable f
      ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g (tsum fun i => f i))
    -/
  · exact tsum_const_smul g hf
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g (tsum fun i => f i))
  -/
  rw [tsum_eq_zero_of_not_summable hf]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g 0)
  -/
  simp only [smul_zero]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) 0
  -/
  let mul_g : α ≃+ α := DistribMulAction.toAddEquiv α g
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    mul_g : AddEquiv α α := DistribMulAction.toAddEquiv α g
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) 0
  -/
  apply tsum_eq_zero_of_not_summable
  /-
    case neg.h
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    mul_g : AddEquiv α α := DistribMulAction.toAddEquiv α g
    ⊢ Not (Summable fun b => HSMul.hSMul g (f b))
  -/
  change ¬ Summable (mul_g ∘ f)
  /-
    case neg.h
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : Group γ
    inst✝² : DistribMulAction γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    hf : Not (Summable f)
    mul_g : AddEquiv α α := DistribMulAction.toAddEquiv α g
    ⊢ Not (Summable (Function.comp (⇑mul_g) f))
  -/
  rwa [Summable.map_iff_of_equiv mul_g]
    /-
      case neg.h.hg
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : AddCommMonoid α
      f : β → α
      γ : Type u_5
      inst✝³ : Group γ
      inst✝² : DistribMulAction γ α
      inst✝¹ : ContinuousConstSMul γ α
      inst✝ : T2Space α
      g : γ
      hf : Not (Summable f)
      mul_g : AddEquiv α α := DistribMulAction.toAddEquiv α g
      ⊢ Continuous ⇑mul_g
    -/
  · apply continuous_const_smul
    /-
      🎉 no goals
    -/
    /-
      case neg.h.hg'
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : AddCommMonoid α
      f : β → α
      γ : Type u_5
      inst✝³ : Group γ
      inst✝² : DistribMulAction γ α
      inst✝¹ : ContinuousConstSMul γ α
      inst✝ : T2Space α
      g : γ
      hf : Not (Summable f)
      mul_g : AddEquiv α α := DistribMulAction.toAddEquiv α g
      ⊢ Continuous (EquivLike.inv mul_g)
    -/
  · apply continuous_const_smul
    /-
      🎉 no goals
    -/


/-- Infinite sums commute with scalar multiplication. Version for scalars living in a
  `DivisionRing`; no summability hypothesis. This could be made to work for a
  `[GroupWithZero γ]` if there was such a thing as `DistribMulActionWithZero`. -/
lemma tsum_const_smul'' {γ : Type*} [DivisionRing γ] [Module γ α] [ContinuousConstSMul γ α]
    [T2Space α] (g : γ) : ∑' (i : β), g • f i = g • ∑' (i : β), f i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : AddCommMonoid α
    f : β → α
    γ : Type u_5
    inst✝³ : DivisionRing γ
    inst✝² : Module γ α
    inst✝¹ : ContinuousConstSMul γ α
    inst✝ : T2Space α
    g : γ
    ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g (tsum fun i => f i))
  -/
  rcases eq_or_ne g 0 with rfl | hg
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : AddCommMonoid α
      f : β → α
      γ : Type u_5
      inst✝³ : DivisionRing γ
      inst✝² : Module γ α
      inst✝¹ : ContinuousConstSMul γ α
      inst✝ : T2Space α
      ⊢ Eq (tsum fun i => HSMul.hSMul 0 (f i)) (HSMul.hSMul 0 (tsum fun i => f i))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : AddCommMonoid α
      f : β → α
      γ : Type u_5
      inst✝³ : DivisionRing γ
      inst✝² : Module γ α
      inst✝¹ : ContinuousConstSMul γ α
      inst✝ : T2Space α
      g : γ
      hg : Ne g 0
      ⊢ Eq (tsum fun i => HSMul.hSMul g (f i)) (HSMul.hSMul g (tsum fun i => f i))
    -/
  · exact tsum_const_smul' (Units.mk0 g hg)
    /-
      🎉 no goals
    -/


theorem HasSum.smul_const {r : R} (hf : HasSum f r) (a : M) : HasSum (fun z ↦ f z • a) (r • a) :=
  hf.map ((smulAddHom R M).flip a) (continuous_id.smul continuous_const)


theorem Summable.smul_const (hf : Summable f) (a : M) : Summable fun z ↦ f z • a :=
  (hf.hasSum.smul_const _).summable


theorem tsum_smul_const [T2Space M] (hf : Summable f) (a : M) : ∑' z, f z • a = (∑' z, f z) • a :=
  (hf.hasSum.smul_const _).tsum_eq


theorem HasSum.smul_eq (hf : HasSum f s) (hg : HasSum g t)
    (hfg : HasSum (fun x : ι × κ ↦ f x.1 • g x.2) u) : s • t = u :=
  have key₁ : HasSum (fun i ↦ f i • t) (s • t) := hf.smul_const t
  have this : ∀ i : ι, HasSum (fun c : κ ↦ f i • g c) (f i • t) := fun i ↦ hg.const_smul (f i)
  have key₂ : HasSum (fun i ↦ f i • t) u := HasSum.prod_fiberwise hfg this
  key₁.unique key₂


theorem HasSum.smul (hf : HasSum f s) (hg : HasSum g t)
    (hfg : Summable fun x : ι × κ ↦ f x.1 • g x.2) :
    HasSum (fun x : ι × κ ↦ f x.1 • g x.2) (s • t) :=
  let ⟨_u, hu⟩ := hfg
  (hf.smul_eq hg hu).symm ▸ hu


/-- Scalar product of two infinites sums indexed by arbitrary types. -/
theorem tsum_smul_tsum (hf : Summable f) (hg : Summable g)
    (hfg : Summable fun x : ι × κ ↦ f x.1 • g x.2) :
    ((∑' x, f x) • ∑' y, g y) = ∑' z : ι × κ, f z.1 • g z.2 :=
  hf.hasSum.smul_eq hg.hasSum hfg.hasSum


/-- Applying a continuous linear map commutes with taking an (infinite) sum. -/
protected theorem ContinuousLinearMap.hasSum {f : ι → M} (φ : M →SL[σ] M₂) {x : M}
    (hf : HasSum f x) : HasSum (fun b : ι ↦ φ (f b)) (φ x) := by
  /-
    ι : Type u_5
    R : Type u_7
    R₂ : Type u_8
    M : Type u_9
    M₂ : Type u_10
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalSpace M₂
    σ : RingHom R R₂
    f : ι → M
    φ : ContinuousLinearMap σ M M₂
    x : M
    hf : HasSum f x
    ⊢ HasSum (fun b => φ (f b)) (φ x)
  -/
  simpa only using hf.map φ.toLinearMap.toAddMonoidHom φ.continuous
  /-
    🎉 no goals
  -/


alias HasSum.mapL := ContinuousLinearMap.hasSum


protected theorem ContinuousLinearMap.summable {f : ι → M} (φ : M →SL[σ] M₂) (hf : Summable f) :
    Summable fun b : ι ↦ φ (f b) :=
  (hf.hasSum.mapL φ).summable


alias Summable.mapL := ContinuousLinearMap.summable


protected theorem ContinuousLinearMap.map_tsum [T2Space M₂] {f : ι → M} (φ : M →SL[σ] M₂)
    (hf : Summable f) : φ (∑' z, f z) = ∑' z, φ (f z) :=
  (hf.hasSum.mapL φ).tsum_eq.symm


/-- Applying a continuous linear map commutes with taking an (infinite) sum. -/
protected theorem ContinuousLinearEquiv.hasSum {f : ι → M} (e : M ≃SL[σ] M₂) {y : M₂} :
    HasSum (fun b : ι ↦ e (f b)) y ↔ HasSum f (e.symm y) :=
              /-
                ι : Type u_5
                R : Type u_7
                R₂ : Type u_8
                M : Type u_9
                M₂ : Type u_10
                inst✝⁹ : Semiring R
                inst✝⁸ : Semiring R₂
                inst✝⁷ : AddCommMonoid M
                inst✝⁶ : Module R M
                inst✝⁵ : AddCommMonoid M₂
                inst✝⁴ : Module R₂ M₂
                inst✝³ : TopologicalSpace M
                inst✝² : TopologicalSpace M₂
                σ : RingHom R R₂
                σ' : RingHom R₂ R
                inst✝¹ : RingHomInvPair σ σ'
                inst✝ : RingHomInvPair σ' σ
                f : ι → M
                e : ContinuousLinearEquiv σ M M₂
                y : M₂
                h : HasSum (fun b => e (f b)) y
                ⊢ HasSum f (e.symm y)
              -/
  ⟨fun h ↦ by simpa only [e.symm.coe_coe, e.symm_apply_apply] using h.mapL (e.symm : M₂ →SL[σ'] M),
              /-
                🎉 no goals
              -/
               /-
                 ι : Type u_5
                 R : Type u_7
                 R₂ : Type u_8
                 M : Type u_9
                 M₂ : Type u_10
                 inst✝⁹ : Semiring R
                 inst✝⁸ : Semiring R₂
                 inst✝⁷ : AddCommMonoid M
                 inst✝⁶ : Module R M
                 inst✝⁵ : AddCommMonoid M₂
                 inst✝⁴ : Module R₂ M₂
                 inst✝³ : TopologicalSpace M
                 inst✝² : TopologicalSpace M₂
                 σ : RingHom R R₂
                 σ' : RingHom R₂ R
                 inst✝¹ : RingHomInvPair σ σ'
                 inst✝ : RingHomInvPair σ' σ
                 f : ι → M
                 e : ContinuousLinearEquiv σ M M₂
                 y : M₂
                 h : HasSum f (e.symm y)
                 ⊢ HasSum (fun b => e (f b)) y
               -/
    fun h ↦ by simpa only [e.coe_coe, e.apply_symm_apply] using (e : M →SL[σ] M₂).hasSum h⟩
               /-
                 🎉 no goals
               -/


/-- Applying a continuous linear map commutes with taking an (infinite) sum. -/
protected theorem ContinuousLinearEquiv.hasSum' {f : ι → M} (e : M ≃SL[σ] M₂) {x : M} :
    HasSum (fun b : ι ↦ e (f b)) (e x) ↔ HasSum f x := by
  /-
    ι : Type u_5
    R : Type u_7
    R₂ : Type u_8
    M : Type u_9
    M₂ : Type u_10
    inst✝⁹ : Semiring R
    inst✝⁸ : Semiring R₂
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R₂ M₂
    inst✝³ : TopologicalSpace M
    inst✝² : TopologicalSpace M₂
    σ : RingHom R R₂
    σ' : RingHom R₂ R
    inst✝¹ : RingHomInvPair σ σ'
    inst✝ : RingHomInvPair σ' σ
    f : ι → M
    e : ContinuousLinearEquiv σ M M₂
    x : M
    ⊢ Iff (HasSum (fun b => e (f b)) (e x)) (HasSum f x)
  -/
  rw [e.hasSum, ContinuousLinearEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


protected theorem ContinuousLinearEquiv.summable {f : ι → M} (e : M ≃SL[σ] M₂) :
    (Summable fun b : ι ↦ e (f b)) ↔ Summable f :=
  ⟨fun hf ↦ (e.hasSum.1 hf.hasSum).summable, (e : M →SL[σ] M₂).summable⟩


theorem ContinuousLinearEquiv.tsum_eq_iff [T2Space M] [T2Space M₂] {f : ι → M} (e : M ≃SL[σ] M₂)
    {y : M₂} : (∑' z, e (f z)) = y ↔ ∑' z, f z = e.symm y := by
  /-
    ι : Type u_5
    R : Type u_7
    R₂ : Type u_8
    M : Type u_9
    M₂ : Type u_10
    inst✝¹¹ : Semiring R
    inst✝¹⁰ : Semiring R₂
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : Module R₂ M₂
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : TopologicalSpace M₂
    σ : RingHom R R₂
    σ' : RingHom R₂ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : T2Space M
    inst✝ : T2Space M₂
    f : ι → M
    e : ContinuousLinearEquiv σ M M₂
    y : M₂
    ⊢ Iff (Eq (tsum fun z => e (f z)) y) (Eq (tsum fun z => f z) (e.symm y))
  -/
  by_cases hf : Summable f
  · exact
      ⟨fun h ↦ (e.hasSum.mp ((e.summable.mpr hf).hasSum_iff.mpr h)).tsum_eq, fun h ↦
        (e.hasSum.mpr (hf.hasSum_iff.mpr h)).tsum_eq⟩
    /-
      case neg
      ι : Type u_5
      R : Type u_7
      R₂ : Type u_8
      M : Type u_9
      M₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : TopologicalSpace M₂
      σ : RingHom R R₂
      σ' : RingHom R₂ R
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomInvPair σ' σ
      inst✝¹ : T2Space M
      inst✝ : T2Space M₂
      f : ι → M
      e : ContinuousLinearEquiv σ M M₂
      y : M₂
      hf : Not (Summable f)
      ⊢ Iff (Eq (tsum fun z => e (f z)) y) (Eq (tsum fun z => f z) (e.symm y))
    -/
  · have hf' : ¬Summable fun z ↦ e (f z) := fun h ↦ hf (e.summable.mp h)
    /-
      case neg
      ι : Type u_5
      R : Type u_7
      R₂ : Type u_8
      M : Type u_9
      M₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : TopologicalSpace M₂
      σ : RingHom R R₂
      σ' : RingHom R₂ R
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomInvPair σ' σ
      inst✝¹ : T2Space M
      inst✝ : T2Space M₂
      f : ι → M
      e : ContinuousLinearEquiv σ M M₂
      y : M₂
      hf : Not (Summable f)
      hf' : Not (Summable fun z => e (f z))
      ⊢ Iff (Eq (tsum fun z => e (f z)) y) (Eq (tsum fun z => f z) (e.symm y))
    -/
    rw [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable hf']
    /-
      case neg
      ι : Type u_5
      R : Type u_7
      R₂ : Type u_8
      M : Type u_9
      M₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : TopologicalSpace M₂
      σ : RingHom R R₂
      σ' : RingHom R₂ R
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomInvPair σ' σ
      inst✝¹ : T2Space M
      inst✝ : T2Space M₂
      f : ι → M
      e : ContinuousLinearEquiv σ M M₂
      y : M₂
      hf : Not (Summable f)
      hf' : Not (Summable fun z => e (f z))
      ⊢ Iff (Eq 0 y) (Eq 0 (e.symm y))
    -/
    refine ⟨?_, fun H ↦ ?_⟩
      /-
        case neg.refine_1
        ι : Type u_5
        R : Type u_7
        R₂ : Type u_8
        M : Type u_9
        M₂ : Type u_10
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : Semiring R₂
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : Module R M
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : TopologicalSpace M₂
        σ : RingHom R R₂
        σ' : RingHom R₂ R
        inst✝³ : RingHomInvPair σ σ'
        inst✝² : RingHomInvPair σ' σ
        inst✝¹ : T2Space M
        inst✝ : T2Space M₂
        f : ι → M
        e : ContinuousLinearEquiv σ M M₂
        y : M₂
        hf : Not (Summable f)
        hf' : Not (Summable fun z => e (f z))
        ⊢ Eq 0 y → Eq 0 (e.symm y)
      -/
    · rintro rfl
      /-
        case neg.refine_1
        ι : Type u_5
        R : Type u_7
        R₂ : Type u_8
        M : Type u_9
        M₂ : Type u_10
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : Semiring R₂
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : Module R M
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : TopologicalSpace M₂
        σ : RingHom R R₂
        σ' : RingHom R₂ R
        inst✝³ : RingHomInvPair σ σ'
        inst✝² : RingHomInvPair σ' σ
        inst✝¹ : T2Space M
        inst✝ : T2Space M₂
        f : ι → M
        e : ContinuousLinearEquiv σ M M₂
        hf : Not (Summable f)
        hf' : Not (Summable fun z => e (f z))
        ⊢ Eq 0 (e.symm 0)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        ι : Type u_5
        R : Type u_7
        R₂ : Type u_8
        M : Type u_9
        M₂ : Type u_10
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : Semiring R₂
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : Module R M
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : TopologicalSpace M₂
        σ : RingHom R R₂
        σ' : RingHom R₂ R
        inst✝³ : RingHomInvPair σ σ'
        inst✝² : RingHomInvPair σ' σ
        inst✝¹ : T2Space M
        inst✝ : T2Space M₂
        f : ι → M
        e : ContinuousLinearEquiv σ M M₂
        y : M₂
        hf : Not (Summable f)
        hf' : Not (Summable fun z => e (f z))
        H : Eq 0 (e.symm y)
        ⊢ Eq 0 y
      -/
    · simpa using congr_arg (fun z ↦ e z) H
      /-
        🎉 no goals
      -/


protected theorem ContinuousLinearEquiv.map_tsum [T2Space M] [T2Space M₂] {f : ι → M}
    (e : M ≃SL[σ] M₂) : e (∑' z, f z) = ∑' z, e (f z) := by
  /-
    ι : Type u_5
    R : Type u_7
    R₂ : Type u_8
    M : Type u_9
    M₂ : Type u_10
    inst✝¹¹ : Semiring R
    inst✝¹⁰ : Semiring R₂
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : Module R₂ M₂
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : TopologicalSpace M₂
    σ : RingHom R R₂
    σ' : RingHom R₂ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : T2Space M
    inst✝ : T2Space M₂
    f : ι → M
    e : ContinuousLinearEquiv σ M M₂
    ⊢ Eq (e (tsum fun z => f z)) (tsum fun z => e (f z))
  -/
  refine symm (e.tsum_eq_iff.mpr ?_)
  /-
    ι : Type u_5
    R : Type u_7
    R₂ : Type u_8
    M : Type u_9
    M₂ : Type u_10
    inst✝¹¹ : Semiring R
    inst✝¹⁰ : Semiring R₂
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : Module R₂ M₂
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : TopologicalSpace M₂
    σ : RingHom R R₂
    σ' : RingHom R₂ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : T2Space M
    inst✝ : T2Space M₂
    f : ι → M
    e : ContinuousLinearEquiv σ M M₂
    ⊢ Eq (tsum fun z => f z) (e.symm (e (tsum fun z => f z)))
  -/
  rw [e.symm_apply_apply _]
  /-
    🎉 no goals
  -/


/-- Given a group `α` acting on a type `β`, and a function `f : β → M`, we "automorphize" `f` to a
  function `β ⧸ α → M` by summing over `α` orbits, `b ↦ ∑' (a : α), f(a • b)`. -/
@[to_additive "Given an additive group `α` acting on a type `β`, and a function `f : β → M`,
  we automorphize `f` to a function `β ⧸ α → M` by summing over `α` orbits,
  `b ↦ ∑' (a : α), f(a • b)`."]
noncomputable def MulAction.automorphize [Group α] [MulAction α β] (f : β → M) :
    Quotient (MulAction.orbitRel α β) → M := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    ⊢ Quotient (MulAction.orbitRel α β) → M
  -/
  refine @Quotient.lift _ _ (_) (fun b ↦ ∑' (a : α), f (a • b)) ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    ⊢ ∀ (a b : β), HasEquiv.Equiv a b → Eq ((fun b => tsum fun a => f (HSMul.hSMul …
  -/
  intro b₁ b₂ ⟨a, (ha : a • b₂ = b₁)⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq ((fun b => tsum fun a => f (HSMul.hSMul a b)) b₁) ((fun b => tsum fun a = …
  -/
  simp only
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq (tsum fun a => f (HSMul.hSMul a b₁)) (tsum fun a => f (HSMul.hSMul a b₂))
  -/
  rw [← ha]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq (tsum fun a_1 => f (HSMul.hSMul a_1 (HSMul.hSMul a b₂))) (tsum fun a => f …
  -/
  convert (Equiv.mulRight a).tsum_eq (fun a' ↦ f (a' • b₂)) using 1
  /-
    case h.e'_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq (tsum fun a_1 => f (HSMul.hSMul a_1 (HSMul.hSMul a b₂))) (tsum fun c => f …
  -/
  simp only [Equiv.coe_mulRight]
  /-
    case h.e'_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq (tsum fun a_1 => f (HSMul.hSMul a_1 (HSMul.hSMul a b₂))) (tsum fun c => f …
  -/
  congr
  /-
    case h.e'_2.e_f
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    ⊢ Eq (fun a_1 => f (HSMul.hSMul a_1 (HSMul.hSMul a b₂))) fun c => f (HSMul.hSM …
  -/
  ext
  /-
    case h.e'_2.e_f.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    x✝ : α
    ⊢ Eq (f (HSMul.hSMul x✝ (HSMul.hSMul a b₂))) (f (HSMul.hSMul (HMul.hMul x✝ a)  …
  -/
  congr 1
  /-
    case h.e'_2.e_f.h.e_a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Type u_5
    κ : Type u_6
    R✝ : Type u_7
    R₂ : Type u_8
    M✝ : Type u_9
    M₂ : Type u_10
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    b₁ b₂ : β
    a : α
    ha : Eq (HSMul.hSMul a b₂) b₁
    x✝ : α
    ⊢ Eq (HSMul.hSMul x✝ (HSMul.hSMul a b₂)) (HSMul.hSMul (HMul.hMul x✝ a) b₂)
  -/
  simp only [mul_smul]
  /-
    🎉 no goals
  -/


/-- Automorphization of a function into an `R`-`Module` distributes, that is, commutes with the
`R`-scalar multiplication. -/
lemma MulAction.automorphize_smul_left [Group α] [MulAction α β] (f : β → M)
    (g : Quotient (MulAction.orbitRel α β) → R) :
    MulAction.automorphize ((g ∘ (@Quotient.mk' _ (_))) • f)
      = g • (MulAction.automorphize f : Quotient (MulAction.orbitRel α β) → M) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    ⊢ Eq (MulAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f)) ( …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    ⊢ Eq (MulAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f) x) …
  -/
  apply @Quotient.inductionOn' β (MulAction.orbitRel α β) _ x _
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    ⊢ ∀ (a : β), Eq (MulAction.automorphize (HSMul.hSMul (Function.comp g Quotient …
  -/
  intro b
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    b : β
    ⊢ Eq (MulAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f) (Q …
  -/
  simp only [automorphize, Pi.smul_apply', comp_apply]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    b : β
    ⊢ Eq (Quotient.lift (fun b => tsum fun a => HSMul.hSMul (g (Quotient.mk' (HSMu …
  -/
  set π : β → Quotient (MulAction.orbitRel α β) := Quotient.mk (MulAction.orbitRel α β)
  have H₁ : ∀ a : α, π (a • b) = π b := by
    intro a
    apply (@Quotient.eq _ (MulAction.orbitRel α β) (a • b) b).mpr
    use a
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    b : β
    π : β → Quotient (MulAction.orbitRel α β) := Quotient.mk (MulAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HSMul.hSMul a b)) (π b)
    ⊢ Eq (Quotient.lift (fun b => tsum fun a => HSMul.hSMul (g (Quotient.mk' (HSMu …
  -/
  change ∑' a : α, g (π (a • b)) • f (a • b) = g (π b) • ∑' a : α, f (a • b)
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    b : β
    π : β → Quotient (MulAction.orbitRel α β) := Quotient.mk (MulAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HSMul.hSMul a b)) (π b)
    ⊢ Eq (tsum fun a => HSMul.hSMul (g (π (HSMul.hSMul a b))) (f (HSMul.hSMul a b) …
  -/
  simp_rw [H₁]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : Group α
    inst✝ : MulAction α β
    f : β → M
    g : Quotient (MulAction.orbitRel α β) → R
    x : Quotient (MulAction.orbitRel α β)
    b : β
    π : β → Quotient (MulAction.orbitRel α β) := Quotient.mk (MulAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HSMul.hSMul a b)) (π b)
    ⊢ Eq (tsum fun a => HSMul.hSMul (g (π b)) (f (HSMul.hSMul a b))) (HSMul.hSMul  …
  -/
  exact tsum_const_smul'' _
  /-
    🎉 no goals
  -/


/-- Automorphization of a function into an `R`-`Module` distributes, that is, commutes with the
`R`-scalar multiplication. -/
lemma AddAction.automorphize_smul_left [AddGroup α] [AddAction α β]  (f : β → M)
    (g : Quotient (AddAction.orbitRel α β) → R) :
    AddAction.automorphize ((g ∘ (@Quotient.mk' _ (_))) • f)
      = g • (AddAction.automorphize f : Quotient (AddAction.orbitRel α β) → M) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    ⊢ Eq (AddAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f)) ( …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    ⊢ Eq (AddAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f) x) …
  -/
  apply @Quotient.inductionOn' β (AddAction.orbitRel α β) _ x _
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    ⊢ ∀ (a : β), Eq (AddAction.automorphize (HSMul.hSMul (Function.comp g Quotient …
  -/
  intro b
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    b : β
    ⊢ Eq (AddAction.automorphize (HSMul.hSMul (Function.comp g Quotient.mk') f) (Q …
  -/
  simp only [automorphize, Pi.smul_apply', comp_apply]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    b : β
    ⊢ Eq (Quotient.lift (fun b => tsum fun a => HSMul.hSMul (g (Quotient.mk' (HVAd …
  -/
  set π : β → Quotient (AddAction.orbitRel α β) := Quotient.mk (AddAction.orbitRel α β)
  have H₁ : ∀ a : α, π (a +ᵥ b) = π b := by
    intro a
    apply (@Quotient.eq _ (AddAction.orbitRel α β) (a +ᵥ b) b).mpr
    use a
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    b : β
    π : β → Quotient (AddAction.orbitRel α β) := Quotient.mk (AddAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HVAdd.hVAdd a b)) (π b)
    ⊢ Eq (Quotient.lift (fun b => tsum fun a => HSMul.hSMul (g (Quotient.mk' (HVAd …
  -/
  change ∑' a : α, g (π (a +ᵥ b)) • f (a +ᵥ b) = g (π b) • ∑' a : α, f (a +ᵥ b)
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    b : β
    π : β → Quotient (AddAction.orbitRel α β) := Quotient.mk (AddAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HVAdd.hVAdd a b)) (π b)
    ⊢ Eq (tsum fun a => HSMul.hSMul (g (π (HVAdd.hVAdd a b))) (f (HVAdd.hVAdd a b) …
  -/
  simp_rw [H₁]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_11
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : T2Space M
    R : Type u_12
    inst✝⁴ : DivisionRing R
    inst✝³ : Module R M
    inst✝² : ContinuousConstSMul R M
    inst✝¹ : AddGroup α
    inst✝ : AddAction α β
    f : β → M
    g : Quotient (AddAction.orbitRel α β) → R
    x : Quotient (AddAction.orbitRel α β)
    b : β
    π : β → Quotient (AddAction.orbitRel α β) := Quotient.mk (AddAction.orbitRel α …
    H₁ : ∀ (a : α), Eq (π (HVAdd.hVAdd a b)) (π b)
    ⊢ Eq (tsum fun a => HSMul.hSMul (g (π b)) (f (HVAdd.hVAdd a b))) (HSMul.hSMul  …
  -/
  exact tsum_const_smul'' _
  /-
    🎉 no goals
  -/


/-- Given a subgroup `Γ` of a group `G`, and a function `f : G → M`, we "automorphize" `f` to a
  function `G ⧸ Γ → M` by summing over `Γ` orbits, `g ↦ ∑' (γ : Γ), f(γ • g)`. -/
@[to_additive "Given a subgroup `Γ` of an additive group `G`, and a function `f : G → M`, we
  automorphize `f` to a function `G ⧸ Γ → M` by summing over `Γ` orbits,
  `g ↦ ∑' (γ : Γ), f(γ • g)`."]
noncomputable def QuotientGroup.automorphize (f : G → M) : G ⧸ Γ → M := MulAction.automorphize f


/-- Automorphization of a function into an `R`-`Module` distributes, that is, commutes with the
`R`-scalar multiplication. -/
lemma QuotientGroup.automorphize_smul_left (f : G → M) (g : G ⧸ Γ → R) :
    (QuotientGroup.automorphize ((g ∘ (@Quotient.mk' _ (_)) : G → R) • f) : G ⧸ Γ → M)
      = g • (QuotientGroup.automorphize f : G ⧸ Γ → M) :=
  MulAction.automorphize_smul_left f g


/-- Automorphization of a function into an `R`-`Module` distributes, that is, commutes with the
`R`-scalar multiplication. -/
lemma QuotientAddGroup.automorphize_smul_left (f : G → M) (g : G ⧸ Γ → R) :
    QuotientAddGroup.automorphize ((g ∘ (@Quotient.mk' _ (_))) • f)
      = g • (QuotientAddGroup.automorphize f : G ⧸ Γ → M) :=
  AddAction.automorphize_smul_left f g



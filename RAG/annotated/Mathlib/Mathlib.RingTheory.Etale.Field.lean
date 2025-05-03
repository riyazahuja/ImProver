/--
This is a weaker version of `of_isSeparable` that additionally assumes `EssFiniteType K L`.
Use that instead.

This is Iversen Corollary II.5.3.
-/
theorem of_isSeparable_aux [Algebra.IsSeparable K L] [EssFiniteType K L] :
    FormallyEtale K L := by
  -- We already know that for field extensions
  -- IsSeparable + EssFiniteType => FormallyUnramified + Finite
  /-
    K L : Type u
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Algebra.EssFiniteType K L
    ⊢ Algebra.FormallyEtale K L
  -/
  have := FormallyUnramified.of_isSeparable K L
  /-
    K L : Type u
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Algebra.EssFiniteType K L
    this : Algebra.FormallyUnramified K L
    ⊢ Algebra.FormallyEtale K L
  -/
  have := FormallyUnramified.finite_of_free (R := K) (S := L)
  /-
    K L : Type u
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    ⊢ Algebra.FormallyEtale K L
  -/
  constructor
  -- We shall show that any `f : L → B/I` can be lifted to `L → B` if `I^2 = ⊥`
  /-
    case comp_bijective
    K L : Type u
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra K B] (I : Ideal B), Eq  …
  -/
  intros B _ _ I h
  /-
    case comp_bijective
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    ⊢ Function.Bijective (Ideal.Quotient.mkₐ K I).comp
  -/
  refine ⟨FormallyUnramified.iff_comp_injective.mp (FormallyUnramified.of_isSeparable K L) I h, ?_⟩
  /-
    case comp_bijective
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    ⊢ Function.Surjective (Ideal.Quotient.mkₐ K I).comp
  -/
  intro f
  -- By separability and finiteness, we may assume `L = K(α)` with `p` the minpoly of `α`.
  /-
    case comp_bijective
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  let pb := Field.powerBasisOfFiniteOfSeparable K L
  -- Let `x : B` such that `f(α) = x` in `B / I`.
  /-
    case comp_bijective
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  obtain ⟨x, hx⟩ := Ideal.Quotient.mk_surjective (f pb.gen)
  /-
    case comp_bijective.intro
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    x : B
    hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  have helper : ∀ x, IsScalarTower.toAlgHom K B (B ⧸ I) x = Ideal.Quotient.mk I x := fun _ ↦ rfl
  -- Then `p(x) = 0 mod I`, and the goal is to find some `ε ∈ I` such that
  -- `p(x + ε) = p(x) + ε p'(x) = 0`, and we will get our lift into `B`.
  have hx' : Ideal.Quotient.mk I (aeval x (minpoly K pb.gen)) = 0 := by
    rw [← helper, ← aeval_algHom_apply, helper, hx, aeval_algHom_apply, minpoly.aeval, map_zero]
  -- Since `p` is separable, `-p'(x)` is invertible in `B ⧸ I`,
  obtain ⟨u, hu⟩ : ∃ u, (aeval x) (derivative (minpoly K pb.gen)) * u + 1 ∈ I := by
    have := (isUnit_iff_ne_zero.mpr ((Algebra.IsSeparable.isSeparable K
      pb.gen).aeval_derivative_ne_zero (minpoly.aeval K _))).map f
    rw [← aeval_algHom_apply, ← hx, ← helper, aeval_algHom_apply, helper] at this
    obtain ⟨u, hu⟩ := Ideal.Quotient.mk_surjective (-this.unit⁻¹ : B ⧸ I)
    use u
    rw [← Ideal.Quotient.eq_zero_iff_mem, map_add, map_mul, map_one, hu, mul_neg,
      IsUnit.mul_val_inv, neg_add_cancel]
  -- And `ε = p(x)/(-p'(x))` works.
  /-
    case comp_bijective.intro.intro
    K L : Type u
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra.IsSeparable K L
    inst✝² : Algebra.EssFiniteType K L
    this✝ : Algebra.FormallyUnramified K L
    this : Module.Finite K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    x : B
    hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
    helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
    hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
    u : B
    hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  use pb.liftEquiv.symm ⟨x + u * aeval x (minpoly K pb.gen), ?_⟩
    /-
      case h
      K L : Type u
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : Algebra.EssFiniteType K L
      this✝ : Algebra.FormallyUnramified K L
      this : Module.Finite K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
      x : B
      hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
      helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
      hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
      u : B
      hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
      ⊢ Eq ((Ideal.Quotient.mkₐ K I).comp (pb.liftEquiv.symm ⟨HAdd.hAdd x (HMul.hMul …
    -/
  · apply pb.algHom_ext
    /-
      case h.h
      K L : Type u
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : Algebra.EssFiniteType K L
      this✝ : Algebra.FormallyUnramified K L
      this : Module.Finite K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
      x : B
      hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
      helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
      hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
      u : B
      hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
      ⊢ Eq (((Ideal.Quotient.mkₐ K I).comp (pb.liftEquiv.symm ⟨HAdd.hAdd x (HMul.hMu …
    -/
    simp [hx, hx']
    /-
      🎉 no goals
    -/
  · rw [← eval_map_algebraMap, Polynomial.eval_add_of_sq_eq_zero, derivative_map,
      ← one_mul (eval x _), eval_map_algebraMap, eval_map_algebraMap, ← mul_assoc, ← add_mul,
      ← Ideal.mem_bot, ← h, pow_two, add_comm]
      /-
        case w
        K L : Type u
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra.IsSeparable K L
        inst✝² : Algebra.EssFiniteType K L
        this✝ : Algebra.FormallyUnramified K L
        this : Module.Finite K L
        B : Type u
        inst✝¹ : CommRing B
        inst✝ : Algebra K B
        I : Ideal B
        h : Eq (HPow.hPow I 2) Bot.bot
        f : AlgHom K L (HasQuotient.Quotient B I)
        pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
        x : B
        hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
        helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
        hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
        u : B
        hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
        ⊢ Membership.mem (HMul.hMul I I) (HMul.hMul (HAdd.hAdd (HMul.hMul ((Polynomial …
      -/
    · exact Ideal.mul_mem_mul hu (Ideal.Quotient.eq_zero_iff_mem.mp hx')
      /-
        🎉 no goals
      -/
    /-
      case w.hy
      K L : Type u
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : Algebra.EssFiniteType K L
      this✝ : Algebra.FormallyUnramified K L
      this : Module.Finite K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
      x : B
      hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
      helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
      hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
      u : B
      hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
      ⊢ Eq (HPow.hPow (HMul.hMul u ((Polynomial.aeval x) (minpoly K pb.gen))) 2) 0
    -/
    rw [← Ideal.mem_bot, ← h]
    /-
      case w.hy
      K L : Type u
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : Algebra.EssFiniteType K L
      this✝ : Algebra.FormallyUnramified K L
      this : Module.Finite K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
      x : B
      hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
      helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
      hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
      u : B
      hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
      ⊢ Membership.mem (HPow.hPow I 2) (HPow.hPow (HMul.hMul u ((Polynomial.aeval x) …
    -/
    apply Ideal.pow_mem_pow
    /-
      case w.hy.hx
      K L : Type u
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra.IsSeparable K L
      inst✝² : Algebra.EssFiniteType K L
      this✝ : Algebra.FormallyUnramified K L
      this : Module.Finite K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
      x : B
      hx : Eq ((Ideal.Quotient.mk I) x) (f pb.gen)
      helper : ∀ (x : B), Eq ((IsScalarTower.toAlgHom K B (HasQuotient.Quotient B I) …
      hx' : Eq ((Ideal.Quotient.mk I) ((Polynomial.aeval x) (minpoly K pb.gen))) 0
      u : B
      hu : Membership.mem I (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) (Polynomial. …
      ⊢ Membership.mem I (HMul.hMul u ((Polynomial.aeval x) (minpoly K pb.gen)))
    -/
    rw [← Ideal.Quotient.eq_zero_iff_mem, map_mul, hx', mul_zero]
    /-
      🎉 no goals
    -/


open scoped IntermediateField in
lemma of_isSeparable [Algebra.IsSeparable K L] : FormallyEtale K L := by
  /-
    K L : Type u
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Algebra.FormallyEtale K L
  -/
  constructor
  /-
    case comp_bijective
    K L : Type u
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra K B] (I : Ideal B), Eq  …
  -/
  intros B _ _ I h
  -- We shall show that any `f : L → B/I` can be lifted to `L → B` if `I^2 = ⊥`.
  -- But we already know that there exists a unique lift for every finite subfield of `L`
  -- by `of_isSeparable_aux`, so we can glue them all together.
  /-
    case comp_bijective
    K L : Type u
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : Algebra.IsSeparable K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    ⊢ Function.Bijective (Ideal.Quotient.mkₐ K I).comp
  -/
  refine ⟨FormallyUnramified.iff_comp_injective.mp (FormallyUnramified.of_isSeparable K L) I h, ?_⟩
  /-
    case comp_bijective
    K L : Type u
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : Algebra.IsSeparable K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    ⊢ Function.Surjective (Ideal.Quotient.mkₐ K I).comp
  -/
  intro f
  have : ∀ k : L, ∃! g : K⟮k⟯ →ₐ[K] B,
      (Ideal.Quotient.mkₐ K I).comp g = f.comp (IsScalarTower.toAlgHom K _ L) := by
    intro k
    have := IsSeparable.of_algHom _ _ (IsScalarTower.toAlgHom K (K⟮k⟯) L)
    have := IntermediateField.adjoin.finiteDimensional
      (Algebra.IsSeparable.isSeparable K k).isIntegral
    have := FormallyEtale.of_isSeparable_aux K (K⟮k⟯)
    have := FormallyEtale.comp_bijective (R := K) (A := K⟮k⟯) I h
    exact this.existsUnique _
  /-
    case comp_bijective
    K L : Type u
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : Algebra.IsSeparable K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    this : ∀ (k : L), ExistsUnique fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) ( …
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  choose g hg₁ hg₂ using this
  have hg₃ : ∀ x y (h : x ∈ K⟮y⟯), g y ⟨x, h⟩ = g x (IntermediateField.AdjoinSimple.gen K x) := by
    intro x y h
    have e : K⟮x⟯ ≤ K⟮y⟯ := by
      rw [IntermediateField.adjoin_le_iff]
      rintro _ rfl
      exact h
    rw [← hg₂ _ ((g _).comp (IntermediateField.inclusion e))]
    · rfl
    apply AlgHom.ext
    intro ⟨a, _⟩
    rw [← AlgHom.comp_assoc, hg₁, AlgHom.comp_assoc]
    simp
  have H : ∀ x y : L, ∃ α : L, x ∈ K⟮α⟯ ∧ y ∈ K⟮α⟯ := by
    intro x y
    have : FiniteDimensional K K⟮x, y⟯ := by
      apply IntermediateField.finiteDimensional_adjoin
      intro x _; exact (Algebra.IsSeparable.isSeparable K x).isIntegral
    have := IsSeparable.of_algHom _ _ (IsScalarTower.toAlgHom K (K⟮x, y⟯) L)
    obtain ⟨⟨α, hα⟩, e⟩ := Field.exists_primitive_element K K⟮x,y⟯
    apply_fun (IntermediateField.map (IntermediateField.val _)) at e
    rw [IntermediateField.adjoin_map, ← AlgHom.fieldRange_eq_map] at e
    simp only [IntermediateField.coe_val, Set.image_singleton,
      IntermediateField.fieldRange_val] at e
    have hx : x ∈ K⟮α⟯ := e ▸ IntermediateField.subset_adjoin K {x, y} (by simp)
    have hy : y ∈ K⟮α⟯ := e ▸ IntermediateField.subset_adjoin K {x, y} (by simp)
    exact ⟨α, hx, hy⟩
  /-
    case comp_bijective
    K L : Type u
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    inst✝² : Algebra.IsSeparable K L
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra K B
    I : Ideal B
    h : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom K L (HasQuotient.Quotient B I)
    g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
    hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
    hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
    hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
    H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ K I).comp a) f
  -/
  refine ⟨⟨⟨⟨⟨fun x ↦ g x (IntermediateField.AdjoinSimple.gen K x), ?_⟩, ?_⟩, ?_, ?_⟩, ?_⟩, ?_⟩
    /-
      case comp_bijective.refine_1
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ Eq ((fun x => (g x) (IntermediateField.AdjoinSimple.gen K x)) 1) 1
    -/
  · show g 1 1 = 1; rw [map_one]
                    /-
                      🎉 no goals
                    -/
    /-
      case comp_bijective.refine_2
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ ∀ (x y : L), Eq ({ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.g …
    -/
  · intros x y
    /-
      case comp_bijective.refine_2
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y : L
      ⊢ Eq ({ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), map_ …
    -/
    obtain ⟨α, hx, hy⟩ := H x y
    /-
      case comp_bijective.refine_2.intro.intro
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y α : L
      hx : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) x
      hy : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) y
      ⊢ Eq ({ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), map_ …
    -/
    simp only [← hg₃ _ _ hx, ← hg₃ _ _ hy, ← map_mul, ← hg₃ _ _ (mul_mem hx hy)]
    /-
      case comp_bijective.refine_2.intro.intro
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y α : L
      hx : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) x
      hy : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) y
      ⊢ Eq ((g α) ⟨HMul.hMul x y, ⋯⟩) ((g α) (HMul.hMul ⟨x, hx⟩ ⟨y, hy⟩))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case comp_bijective.refine_3
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ Eq ((↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), ma …
    -/
  · show g 0 0 = 0; rw [map_zero]
                    /-
                      🎉 no goals
                    -/
    /-
      case comp_bijective.refine_4
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ ∀ (x y : L), Eq ((↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple …
    -/
  · intros x y
    /-
      case comp_bijective.refine_4
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y : L
      ⊢ Eq ((↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), ma …
    -/
    obtain ⟨α, hx, hy⟩ := H x y
    /-
      case comp_bijective.refine_4.intro.intro
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y α : L
      hx : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) x
      hy : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) y
      ⊢ Eq ((↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), ma …
    -/
    simp only [← hg₃ _ _ hx, ← hg₃ _ _ hy, ← map_add, ← hg₃ _ _ (add_mem hx hy)]
    /-
      case comp_bijective.refine_4.intro.intro
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x y α : L
      hx : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) x
      hy : Membership.mem (IntermediateField.adjoin K (Singleton.singleton α)) y
      ⊢ Eq ((g α) ⟨HAdd.hAdd x y, ⋯⟩) ((g α) (HAdd.hAdd ⟨x, hx⟩ ⟨y, hy⟩))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case comp_bijective.refine_5
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ ∀ (r : K), Eq ((↑↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple. …
    -/
  · intro r
    /-
      case comp_bijective.refine_5
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      r : K
      ⊢ Eq ((↑↑{ toFun := fun x => (g x) (IntermediateField.AdjoinSimple.gen K x), m …
    -/
    show g _ (algebraMap K _ r) = _
    /-
      case comp_bijective.refine_5
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      r : K
      ⊢ Eq ((g ((algebraMap K L) r)) ((algebraMap K (Subtype fun x => Membership.mem …
    -/
    rw [AlgHom.commutes]
    /-
      🎉 no goals
    -/
    /-
      case comp_bijective.refine_6
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      ⊢ Eq ((Ideal.Quotient.mkₐ K I).comp { toFun := fun x => (g x) (IntermediateFie …
    -/
  · ext x
    /-
      case comp_bijective.refine_6.H
      K L : Type u
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra.IsSeparable K L
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra K B
      I : Ideal B
      h : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom K L (HasQuotient.Quotient B I)
      g : (k : L) → AlgHom K (Subtype fun x => Membership.mem (IntermediateField.adj …
      hg₁ : ∀ (k : L), (fun g => Eq ((Ideal.Quotient.mkₐ K I).comp g) (f.comp (IsSca …
      hg₂ : ∀ (k : L) (y : AlgHom K (Subtype fun x => Membership.mem (IntermediateFi …
      hg₃ : ∀ (x y : L) (h : Membership.mem (IntermediateField.adjoin K (Singleton.s …
      H : ∀ (x y : L), Exists fun α => And (Membership.mem (IntermediateField.adjoin …
      x : L
      ⊢ Eq (((Ideal.Quotient.mkₐ K I).comp { toFun := fun x => (g x) (IntermediateFi …
    -/
    simpa using AlgHom.congr_fun (hg₁ x) (IntermediateField.AdjoinSimple.gen K x)
    /-
      🎉 no goals
    -/


theorem iff_isSeparable [EssFiniteType K L] :
    FormallyEtale K L ↔ Algebra.IsSeparable K L :=
  ⟨fun _ ↦ FormallyUnramified.isSeparable K L, fun _ ↦ of_isSeparable K L⟩



theorem extendScalars [alg : Algebra.IsAlgebraic R S]
    (inj : Injective (algebraMap S A)) : AlgebraicIndependent S x := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    ⊢ AlgebraicIndependent S x
  -/
  nontriviality S
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    ⊢ AlgebraicIndependent S x
  -/
  have := inj.nontrivial
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this : Nontrivial A
    ⊢ AlgebraicIndependent S x
  -/
  refine algebraicIndependent_of_finite_type' inj fun t fin ind i hi ↦ ?_
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this : Nontrivial A
    t : Set ι
    fin : t.Finite
    ind : AlgebraicIndependent S fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin S (Set.ima …
  -/
  let Rt := adjoin R (x '' t)
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this : Nontrivial A
    t : Set ι
    fin : t.Finite
    ind : AlgebraicIndependent S fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    Rt : Subalgebra R A := Algebra.adjoin R (Set.image x t)
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin S (Set.ima …
  -/
  let St := adjoin S (x '' t)
  let _ : Algebra Rt St :=
    (Rt.inclusion (T := St.restrictScalars R) <| adjoin_le <| by exact subset_adjoin).toAlgebra
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this : Nontrivial A
    t : Set ι
    fin : t.Finite
    ind : AlgebraicIndependent S fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    Rt : Subalgebra R A := Algebra.adjoin R (Set.image x t)
    St : Subalgebra S A := Algebra.adjoin S (Set.image x t)
    x✝ : Algebra (Subtype fun x => Membership.mem Rt x) (Subtype fun x => Membersh …
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin S (Set.ima …
  -/
  have : IsScalarTower Rt St A := .of_algebraMap_eq fun ⟨y, _⟩ ↦ show y = y from rfl
  have : NoZeroDivisors St := (Set.image_eq_range _ _ ▸ ind.aevalEquiv)
    |>.symm.injective.noZeroDivisors _ (map_zero _) (map_mul _)
  have : NoZeroDivisors Rt := (Subalgebra.inclusion_injective _).noZeroDivisors
    (algebraMap Rt St) (map_zero _) (map_mul _)
  have : Algebra.IsAlgebraic Rt St := ⟨fun ⟨y, hy⟩ ↦ by
    rw [← isAlgebraic_algHom_iff (IsScalarTower.toAlgHom Rt St A) Subtype.val_injective]
    show IsAlgebraic Rt y
    exact adjoin_induction (fun _ h ↦ isAlgebraic_algebraMap (⟨_, subset_adjoin h⟩ : Rt))
      (fun z ↦ ((alg.1 z).algHom (IsScalarTower.toAlgHom R S A)).extendScalars fun _ _ eq ↦ by
        exact hx.algebraMap_injective congr($eq.1)) (fun _ _ _ _ ↦ .add) (fun _ _ _ _ ↦ .mul) hy⟩
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this✝³ : Nontrivial A
    t : Set ι
    fin : t.Finite
    ind : AlgebraicIndependent S fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    Rt : Subalgebra R A := Algebra.adjoin R (Set.image x t)
    St : Subalgebra S A := Algebra.adjoin S (Set.image x t)
    x✝ : Algebra (Subtype fun x => Membership.mem Rt x) (Subtype fun x => Membersh …
    this✝² : IsScalarTower (Subtype fun x => Membership.mem Rt x) (Subtype fun x = …
    this✝¹ : NoZeroDivisors (Subtype fun x => Membership.mem St x)
    this✝ : NoZeroDivisors (Subtype fun x => Membership.mem Rt x)
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem Rt x) (Subtype fun …
    ⊢ Transcendental (Subtype fun x_1 => Membership.mem (Algebra.adjoin S (Set.ima …
  -/
  show Transcendental St (x i)
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    alg : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this✝³ : Nontrivial A
    t : Set ι
    fin : t.Finite
    ind : AlgebraicIndependent S fun i => x ↑i
    i : ι
    hi : Not (Membership.mem t i)
    Rt : Subalgebra R A := Algebra.adjoin R (Set.image x t)
    St : Subalgebra S A := Algebra.adjoin S (Set.image x t)
    x✝ : Algebra (Subtype fun x => Membership.mem Rt x) (Subtype fun x => Membersh …
    this✝² : IsScalarTower (Subtype fun x => Membership.mem Rt x) (Subtype fun x = …
    this✝¹ : NoZeroDivisors (Subtype fun x => Membership.mem St x)
    this✝ : NoZeroDivisors (Subtype fun x => Membership.mem Rt x)
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem Rt x) (Subtype fun …
    ⊢ Transcendental (Subtype fun x => Membership.mem St x) (x i)
  -/
  exact (hx.transcendental_adjoin hi).extendScalars Subtype.val_injective
  /-
    🎉 no goals
  -/


theorem extendScalars_of_isSimpleRing [Algebra.IsAlgebraic R S] [IsSimpleRing S] :
    AlgebraicIndependent S x :=
  hx.extendScalars <|
    have := Module.nontrivial R S
    have := hx.algebraMap_injective.nontrivial
    RingHom.injective _


theorem extendScalars_of_isIntegral [Algebra.IsIntegral R S]
    (inj : Injective (algebraMap S A)) : AlgebraicIndependent S x := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    ⊢ AlgebraicIndependent S x
  -/
  nontriviality S
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    ⊢ AlgebraicIndependent S x
  -/
  have := Module.nontrivial R S
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    x : ι → A
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    hx : AlgebraicIndependent R x
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    a✝ : Nontrivial S
    this : Nontrivial R
    ⊢ AlgebraicIndependent S x
  -/
  exact hx.extendScalars inj
  /-
    🎉 no goals
  -/


protected theorem subalgebra (S : Subalgebra R A) [NoZeroDivisors A] [Algebra.IsAlgebraic R S] :
    AlgebraicIndependent S x :=
  hx.extendScalars Subtype.val_injective


theorem subalgebra_of_isIntegral (S : Subalgebra R A) [NoZeroDivisors A] [Algebra.IsIntegral R S] :
    AlgebraicIndependent S x :=
  hx.extendScalars_of_isIntegral Subtype.val_injective


theorem subalgebraAlgebraicClosure [IsDomain R] [NoZeroDivisors A] :
    AlgebraicIndependent (Subalgebra.algebraicClosure R A) x :=
  hx.subalgebra _


protected theorem integralClosure [NoZeroDivisors A] :
    AlgebraicIndependent (integralClosure R A) x :=
  hx.subalgebra_of_isIntegral _


omit hx in
protected theorem algebraicClosure {F E : Type*} [Field F] [Field E] [Algebra F E] {x : ι → E}
    (hx : AlgebraicIndependent F x) : AlgebraicIndependent (algebraicClosure F E) x :=
  hx.extendScalars_of_isSimpleRing



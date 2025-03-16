/-- If `A` and `B` are subalgebras of `S / R`,
then `A` and `B` are linearly disjoint, if they are linearly disjoint as submodules of `S`. -/
protected abbrev LinearDisjoint : Prop := (toSubmodule A).LinearDisjoint (toSubmodule B)


theorem linearDisjoint_iff : A.LinearDisjoint B ↔ (toSubmodule A).LinearDisjoint (toSubmodule B) :=
  Iff.rfl


@[nontriviality]
theorem LinearDisjoint.of_subsingleton [Subsingleton R] : A.LinearDisjoint B :=
  Submodule.LinearDisjoint.of_subsingleton


@[nontriviality]
theorem LinearDisjoint.of_subsingleton_top [Subsingleton S] : A.LinearDisjoint B :=
  Submodule.LinearDisjoint.of_subsingleton_top


/-- Linear disjointness is symmetric if elements in the module commute. -/
theorem LinearDisjoint.symm_of_commute (H : A.LinearDisjoint B)
    (hc : ∀ (a : A) (b : B), Commute a.1 b.1) : B.LinearDisjoint A :=
  Submodule.LinearDisjoint.symm_of_commute H hc


/-- Linear disjointness is symmetric if elements in the module commute. -/
theorem linearDisjoint_comm_of_commute
    (hc : ∀ (a : A) (b : B), Commute a.1 b.1) : A.LinearDisjoint B ↔ B.LinearDisjoint A :=
  ⟨fun H ↦ H.symm_of_commute hc, fun H ↦ H.symm_of_commute fun _ _ ↦ (hc _ _).symm⟩


/-- Linear disjointness is preserved by injective algebra homomorphisms. -/
theorem map (H : A.LinearDisjoint B) {T : Type w} [Semiring T] [Algebra R T]
    (f : S →ₐ[R] T) (hf : Function.Injective f) : (A.map f).LinearDisjoint (B.map f) :=
  Submodule.LinearDisjoint.map H f hf


/-- The image of `R` in `S` is linearly disjoint with any other subalgebras. -/
theorem bot_left : (⊥ : Subalgebra R S).LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    B : Subalgebra R S
    ⊢ Bot.bot.LinearDisjoint B
  -/
  rw [Subalgebra.LinearDisjoint, Algebra.toSubmodule_bot]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    B : Subalgebra R S
    ⊢ Submodule.LinearDisjoint 1 (Subalgebra.toSubmodule B)
  -/
  exact Submodule.LinearDisjoint.one_left _
  /-
    🎉 no goals
  -/


/-- The image of `R` in `S` is linearly disjoint with any other subalgebras. -/
theorem bot_right : A.LinearDisjoint ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    A : Subalgebra R S
    ⊢ A.LinearDisjoint Bot.bot
  -/
  rw [Subalgebra.LinearDisjoint, Algebra.toSubmodule_bot]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    A : Subalgebra R S
    ⊢ (Subalgebra.toSubmodule A).LinearDisjoint 1
  -/
  exact Submodule.LinearDisjoint.one_right _
  /-
    🎉 no goals
  -/


variable (R) in
/-- Images of two `R`-algebras `A` and `B` in `A ⊗[R] B` are linearly disjoint. -/
theorem include_range (A : Type v) [Semiring A] (B : Type w) [Semiring B]
    [Algebra R A] [Algebra R B] :
    (Algebra.TensorProduct.includeLeft : A →ₐ[R] A ⊗[R] B).range.LinearDisjoint
      (Algebra.TensorProduct.includeRight : B →ₐ[R] A ⊗[R] B).range := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    B : Type w
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ Algebra.TensorProduct.includeLeft.range.LinearDisjoint Algebra.TensorProduct …
  -/
  rw [Subalgebra.LinearDisjoint, Submodule.linearDisjoint_iff]
  change Function.Injective <|
    Submodule.mulMap (LinearMap.range Algebra.TensorProduct.includeLeft)
      (LinearMap.range Algebra.TensorProduct.includeRight)
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    B : Type w
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ Function.Injective ⇑((LinearMap.range Algebra.TensorProduct.includeLeft).mul …
  -/
  rw [← Algebra.TensorProduct.linearEquivIncludeRange_symm_toLinearMap]
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    B : Type w
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    ⊢ Function.Injective ⇑↑(Algebra.TensorProduct.linearEquivIncludeRange R A B).s …
  -/
  exact LinearEquiv.injective _
  /-
    🎉 no goals
  -/


/-- Linear disjointness is symmetric in a commutative ring. -/
theorem LinearDisjoint.symm (H : A.LinearDisjoint B) : B.LinearDisjoint A :=
  H.symm_of_commute fun _ _ ↦ mul_comm _ _


/-- Linear disjointness is symmetric in a commutative ring. -/
theorem linearDisjoint_comm : A.LinearDisjoint B ↔ B.LinearDisjoint A :=
  ⟨LinearDisjoint.symm, LinearDisjoint.symm⟩


/-- Two subalgebras `A`, `B` in a commutative ring are linearly disjoint if and only if
`Subalgebra.mulMap A B` is injective. -/
theorem linearDisjoint_iff_injective : A.LinearDisjoint B ↔ Function.Injective (A.mulMap B) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ⊢ Iff (A.LinearDisjoint B) (Function.Injective ⇑(A.mulMap B))
  -/
  rw [linearDisjoint_iff, Submodule.linearDisjoint_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ⊢ Iff (Function.Injective ⇑((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSu …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `A` and `B` are subalgebras in a commutative algebra `S` over `R`, and if they are
linearly disjoint, then there is the natural isomorphism
`A ⊗[R] B ≃ₐ[R] A ⊔ B` induced by multiplication in `S`. -/
protected def mulMap :=
  (AlgEquiv.ofInjective (A.mulMap B) H.injective).trans (equivOfEq _ _ (mulMap_range A B))


@[simp]
theorem val_mulMap_tmul (a : A) (b : B) : (H.mulMap (a ⊗ₜ[R] b) : S) = a.1 * b.1 := rfl


include H in
/-- If `A` and `B` are subalgebras in a commutative algebra `S` over `R`, and if they are
linearly disjoint, and if they are free `R`-modules, then `A ⊔ B` is also a free `R`-module. -/
theorem sup_free_of_free [Module.Free R A] [Module.Free R B] : Module.Free R ↥(A ⊔ B) :=
  Module.Free.of_equiv H.mulMap.toLinearEquiv


include H in
/-- If `A` and `B` are subalgebras in a domain `S` over `R`, and if they are
linearly disjoint, then `A ⊗[R] B` is also a domain. -/
theorem isDomain [IsDomain S] : IsDomain (A ⊗[R] B) :=
  H.injective.isDomain (A.mulMap B).toRingHom


/-- If `A` and `B` are `R`-algebras, such that there exists a domain `S` over `R`
such that `A` and `B` inject into it and their images are linearly disjoint,
then `A ⊗[R] B` is also a domain. -/
theorem isDomain_of_injective [IsDomain S] {A B : Type*} [Semiring A] [Semiring B]
    [Algebra R A] [Algebra R B] {fa : A →ₐ[R] S} {fb : B →ₐ[R] S}
    (hfa : Function.Injective fa) (hfb : Function.Injective fb)
    (H : fa.range.LinearDisjoint fb.range) : IsDomain (A ⊗[R] B) :=
  have := H.isDomain
  (Algebra.TensorProduct.congr
    (AlgEquiv.ofInjective fa hfa) (AlgEquiv.ofInjective fb hfb)).toMulEquiv.isDomain


set_option maxSynthPendingDepth 2 in
lemma mulLeftMap_ker_eq_bot_iff_linearIndependent_op {ι : Type*} (a : ι → A) :
    LinearMap.ker (Submodule.mulLeftMap (M := toSubmodule A) (toSubmodule B) a) = ⊥ ↔
    LinearIndependent B.op (MulOpposite.op ∘ A.val ∘ a) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    ⊢ Iff (Eq (LinearMap.ker (Submodule.mulLeftMap (Subalgebra.toSubmodule B) a))  …
  -/
  simp_rw [LinearIndependent, LinearMap.ker_eq_bot]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    ⊢ Iff (Function.Injective ⇑(Submodule.mulLeftMap (Subalgebra.toSubmodule B) a) …
  -/
  let i : (ι →₀ B) →ₗ[R] S := Submodule.mulLeftMap (M := toSubmodule A) (toSubmodule B) a
  let j : (ι →₀ B) →ₗ[R] S := (MulOpposite.opLinearEquiv _).symm.toLinearMap ∘ₗ
    (Finsupp.linearCombination B.op (MulOpposite.op ∘ A.val ∘ a)).restrictScalars R ∘ₗ
    (Finsupp.mapRange.linearEquiv (linearEquivOp B)).toLinearMap
  suffices i = j by
    change Function.Injective i ↔ _
    simp_rw [this, j, LinearMap.coe_comp, LinearEquiv.coe_coe, EquivLike.comp_injective,
      EquivLike.injective_comp, LinearMap.coe_restrictScalars]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem B x)) …
    j : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem B x)) …
    ⊢ Eq i j
  -/
  ext
  simp only [LinearMap.coe_comp, Function.comp_apply, Finsupp.lsingle_apply, coe_val,
    Finsupp.mapRange.linearEquiv_toLinearMap, LinearEquiv.coe_coe,
    MulOpposite.coe_opLinearEquiv_symm, LinearMap.coe_restrictScalars,
    Finsupp.mapRange.linearMap_apply, Finsupp.mapRange_single, Finsupp.linearCombination_single,
    MulOpposite.unop_smul, MulOpposite.unop_op, i, j]
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem B x)) …
    j : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem B x)) …
    a✝ : ι
    x✝ : Subtype fun x => Membership.mem B x
    ⊢ Eq ((Submodule.mulLeftMap (Subalgebra.toSubmodule B) a) (Finsupp.single a✝ x …
  -/
  exact Submodule.mulLeftMap_apply_single _ _ _
  /-
    🎉 no goals
  -/


variable {A B} in
/-- If `A` and `B` are linearly disjoint, if `B` is a flat `R`-module, then for any family of
`R`-linearly independent elements of `A`, they are also `B`-linearly independent
in the opposite ring. -/
theorem linearIndependent_left_op_of_flat (H : A.LinearDisjoint B) [Module.Flat R B]
    {ι : Type*} {a : ι → A} (ha : LinearIndependent R a) :
    LinearIndependent B.op (MulOpposite.op ∘ A.val ∘ a) := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem B x)
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    ha : LinearIndependent R a
    ⊢ LinearIndependent (Subtype fun x => Membership.mem B.op x) (Function.comp Mu …
  -/
  have h := Submodule.LinearDisjoint.linearIndependent_left_of_flat H ha
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem B x)
    ι : Type u_1
    a : ι → Subtype fun x => Membership.mem A x
    ha : LinearIndependent R a
    h : Eq (LinearMap.ker (Submodule.mulLeftMap (Subalgebra.toSubmodule B) a)) Bot …
    ⊢ LinearIndependent (Subtype fun x => Membership.mem B.op x) (Function.comp Mu …
  -/
  rwa [mulLeftMap_ker_eq_bot_iff_linearIndependent_op] at h
  /-
    🎉 no goals
  -/


/-- If a basis of `A` is also `B`-linearly independent in the opposite ring,
then `A` and `B` are linearly disjoint. -/
theorem of_basis_left_op {ι : Type*} (a : Basis ι R A)
    (H : LinearIndependent B.op (MulOpposite.op ∘ A.val ∘ a)) :
    A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : Basis ι R (Subtype fun x => Membership.mem A x)
    H : LinearIndependent (Subtype fun x => Membership.mem B.op x) (Function.comp  …
    ⊢ A.LinearDisjoint B
  -/
  rw [← mulLeftMap_ker_eq_bot_iff_linearIndependent_op] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    a : Basis ι R (Subtype fun x => Membership.mem A x)
    H : Eq (LinearMap.ker (Submodule.mulLeftMap (Subalgebra.toSubmodule B) ⇑a)) Bo …
    ⊢ A.LinearDisjoint B
  -/
  exact Submodule.LinearDisjoint.of_basis_left _ _ a H
  /-
    🎉 no goals
  -/


set_option maxSynthPendingDepth 2 in
lemma mulRightMap_ker_eq_bot_iff_linearIndependent {ι : Type*} (b : ι → B) :
    LinearMap.ker (Submodule.mulRightMap (toSubmodule A) (N := toSubmodule B) b) = ⊥ ↔
    LinearIndependent A (B.val ∘ b) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    ⊢ Iff (Eq (LinearMap.ker ((Subalgebra.toSubmodule A).mulRightMap b)) Bot.bot)  …
  -/
  simp_rw [LinearIndependent, LinearMap.ker_eq_bot]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    ⊢ Iff (Function.Injective ⇑((Subalgebra.toSubmodule A).mulRightMap b)) (Functi …
  -/
  let i : (ι →₀ A) →ₗ[R] S := Submodule.mulRightMap (toSubmodule A) (N := toSubmodule B) b
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    ⊢ Iff (Function.Injective ⇑((Subalgebra.toSubmodule A).mulRightMap b)) (Functi …
  -/
  let j : (ι →₀ A) →ₗ[R] S := (Finsupp.linearCombination A (B.val ∘ b)).restrictScalars R
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    j : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    ⊢ Iff (Function.Injective ⇑((Subalgebra.toSubmodule A).mulRightMap b)) (Functi …
  -/
  suffices i = j by change Function.Injective i ↔ Function.Injective j; rw [this]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    j : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    ⊢ Eq i j
  -/
  ext
  simp only [LinearMap.coe_comp, Function.comp_apply, Finsupp.lsingle_apply, coe_val,
    LinearMap.coe_restrictScalars, Finsupp.linearCombination_single, i, j]
  /-
    case h.h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    i : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    j : LinearMap (RingHom.id R) (Finsupp ι (Subtype fun x => Membership.mem A x)) …
    a✝ : ι
    x✝ : Subtype fun x => Membership.mem A x
    ⊢ Eq (((Subalgebra.toSubmodule A).mulRightMap b) (Finsupp.single a✝ x✝)) (HSMu …
  -/
  exact Submodule.mulRightMap_apply_single _ _ _
  /-
    🎉 no goals
  -/


variable {A B} in
/-- If `A` and `B` are linearly disjoint, if `A` is a flat `R`-module, then for any family of
`R`-linearly independent elements of `B`, they are also `A`-linearly independent. -/
theorem linearIndependent_right_of_flat (H : A.LinearDisjoint B) [Module.Flat R A]
    {ι : Type*} {b : ι → B} (hb : LinearIndependent R b) :
    LinearIndependent A (B.val ∘ b) := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem A x)
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    hb : LinearIndependent R b
    ⊢ LinearIndependent (Subtype fun x => Membership.mem A x) (Function.comp (⇑B.v …
  -/
  have h := Submodule.LinearDisjoint.linearIndependent_right_of_flat H hb
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem A x)
    ι : Type u_1
    b : ι → Subtype fun x => Membership.mem B x
    hb : LinearIndependent R b
    h : Eq (LinearMap.ker ((Subalgebra.toSubmodule A).mulRightMap b)) Bot.bot
    ⊢ LinearIndependent (Subtype fun x => Membership.mem A x) (Function.comp (⇑B.v …
  -/
  rwa [mulRightMap_ker_eq_bot_iff_linearIndependent] at h
  /-
    🎉 no goals
  -/


/-- If a basis of `B` is also `A`-linearly independent, then `A` and `B` are linearly disjoint. -/
theorem of_basis_right {ι : Type*} (b : Basis ι R B)
    (H : LinearIndependent A (B.val ∘ b)) : A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : Basis ι R (Subtype fun x => Membership.mem B x)
    H : LinearIndependent (Subtype fun x => Membership.mem A x) (Function.comp ⇑B. …
    ⊢ A.LinearDisjoint B
  -/
  rw [← mulRightMap_ker_eq_bot_iff_linearIndependent] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ι : Type u_1
    b : Basis ι R (Subtype fun x => Membership.mem B x)
    H : Eq (LinearMap.ker ((Subalgebra.toSubmodule A).mulRightMap ⇑b)) Bot.bot
    ⊢ A.LinearDisjoint B
  -/
  exact Submodule.LinearDisjoint.of_basis_right _ _ b H
  /-
    🎉 no goals
  -/


variable {A B} in
/-- If `A` and `B` are linearly disjoint and their elements commute, if `B` is a flat `R`-module,
then for any family of `R`-linearly independent elements of `A`,
they are also `B`-linearly independent. -/
theorem linearIndependent_left_of_flat_of_commute (H : A.LinearDisjoint B) [Module.Flat R B]
    {ι : Type*} {a : ι → A} (ha : LinearIndependent R a)
    (hc : ∀ (a : A) (b : B), Commute a.1 b.1) : LinearIndependent B (A.val ∘ a) :=
  (H.symm_of_commute hc).linearIndependent_right_of_flat ha


/-- If a basis of `A` is also `B`-linearly independent, if elements in `A` and `B` commute,
then `A` and `B` are linearly disjoint. -/
theorem of_basis_left_of_commute {ι : Type*} (a : Basis ι R A)
    (H : LinearIndependent B (A.val ∘ a)) (hc : ∀ (a : A) (b : B), Commute a.1 b.1) :
    A.LinearDisjoint B :=
  (of_basis_right B A a H).symm_of_commute fun _ _ ↦ (hc _ _).symm


variable {A B} in
/-- If `A` and `B` are linearly disjoint, if `A` is flat, then for any family of
`R`-linearly independent elements `{ a_i }` of `A`, and any family of
`R`-linearly independent elements `{ b_j }` of `B`, the family `{ a_i * b_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat_left (H : A.LinearDisjoint B) [Module.Flat R A]
    {κ ι : Type*} {a : κ → A} {b : ι → B} (ha : LinearIndependent R a)
    (hb : LinearIndependent R b) : LinearIndependent R fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1 :=
  Submodule.LinearDisjoint.linearIndependent_mul_of_flat_left H ha hb


variable {A B} in
/-- If `A` and `B` are linearly disjoint, if `B` is flat, then for any family of
`R`-linearly independent elements `{ a_i }` of `A`, and any family of
`R`-linearly independent elements `{ b_j }` of `B`, the family `{ a_i * b_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat_right (H : A.LinearDisjoint B) [Module.Flat R B]
    {κ ι : Type*} {a : κ → A} {b : ι → B} (ha : LinearIndependent R a)
    (hb : LinearIndependent R b) : LinearIndependent R fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1 :=
  Submodule.LinearDisjoint.linearIndependent_mul_of_flat_right H ha hb


variable {A B} in
/-- If `A` and `B` are linearly disjoint, if one of `A` and `B` is flat, then for any family of
`R`-linearly independent elements `{ a_i }` of `A`, and any family of
`R`-linearly independent elements `{ b_j }` of `B`, the family `{ a_i * b_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat (H : A.LinearDisjoint B)
    (hf : Module.Flat R A ∨ Module.Flat R B)
    {κ ι : Type*} {a : κ → A} {b : ι → B} (ha : LinearIndependent R a)
    (hb : LinearIndependent R b) : LinearIndependent R fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1 :=
  Submodule.LinearDisjoint.linearIndependent_mul_of_flat H hf ha hb


/-- If `{ a_i }` is an `R`-basis of `A`, if `{ b_j }` is an `R`-basis of `B`,
such that the family `{ a_i * b_j }` in `S` is `R`-linearly independent,
then `A` and `B` are linearly disjoint. -/
theorem of_basis_mul {κ ι : Type*} (a : Basis κ R A) (b : Basis ι R B)
    (H : LinearIndependent R fun (i : κ × ι) ↦ (a i.1).1 * (b i.2).1) : A.LinearDisjoint B :=
  Submodule.LinearDisjoint.of_basis_mul _ _ a b H


theorem of_le_left_of_flat {A' : Subalgebra R S}
    (h : A' ≤ A) [Module.Flat R B] : A'.LinearDisjoint B :=
  Submodule.LinearDisjoint.of_le_left_of_flat H h


theorem of_le_right_of_flat {B' : Subalgebra R S}
    (h : B' ≤ B) [Module.Flat R A] : A.LinearDisjoint B' :=
  Submodule.LinearDisjoint.of_le_right_of_flat H h


theorem of_le_of_flat_right {A' B' : Subalgebra R S}
    (ha : A' ≤ A) (hb : B' ≤ B) [Module.Flat R B] [Module.Flat R A'] :
    A'.LinearDisjoint B' := (H.of_le_left_of_flat ha).of_le_right_of_flat hb


theorem of_le_of_flat_left {A' B' : Subalgebra R S}
    (ha : A' ≤ A) (hb : B' ≤ B) [Module.Flat R A] [Module.Flat R B'] :
    A'.LinearDisjoint B' := (H.of_le_right_of_flat hb).of_le_left_of_flat ha


theorem rank_inf_eq_one_of_commute_of_flat_of_inj (hf : Module.Flat R A ∨ Module.Flat R B)
    (hc : ∀ (a b : ↥(A ⊓ B)), Commute a.1 b.1)
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem A x)) (Module.Flat R ( …
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A B) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Min.min A B) x)) 1
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem A x)) (Module.Flat R ( …
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A B) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    a✝ : Nontrivial R
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Min.min A B) x)) 1
  -/
  refine le_antisymm (Submodule.LinearDisjoint.rank_inf_le_one_of_commute_of_flat H hf hc) ?_
  have : Cardinal.lift.{u} (Module.rank R (⊥ : Subalgebra R S)) =
      Cardinal.lift.{v} (Module.rank R R) :=
    lift_rank_range_of_injective (Algebra.linearMap R S) hinj
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem A x)) (Module.Flat R ( …
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A B) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    a✝ : Nontrivial R
    this : Eq (Cardinal.lift.{u, v} (Module.rank R (Subtype fun x => Membership.me …
    ⊢ LE.le 1 (Module.rank R (Subtype fun x => Membership.mem (Min.min A B) x))
  -/
  rw [Module.rank_self, Cardinal.lift_one, Cardinal.lift_eq_one] at this
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem A x)) (Module.Flat R ( …
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A B) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    a✝ : Nontrivial R
    this : Eq (Module.rank R (Subtype fun x => Membership.mem Bot.bot x)) 1
    ⊢ LE.le 1 (Module.rank R (Subtype fun x => Membership.mem (Min.min A B) x))
  -/
  rw [← this]
  change Module.rank R (toSubmodule (⊥ : Subalgebra R S)) ≤
    Module.rank R (toSubmodule (A ⊓ B))
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem A x)) (Module.Flat R ( …
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A B) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    a✝ : Nontrivial R
    this : Eq (Module.rank R (Subtype fun x => Membership.mem Bot.bot x)) 1
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Subalgebra.toSubmodul …
  -/
  exact Submodule.rank_mono (bot_le : (⊥ : Subalgebra R S) ≤ A ⊓ B)
  /-
    🎉 no goals
  -/


theorem rank_inf_eq_one_of_commute_of_flat_left_of_inj [Module.Flat R A]
    (hc : ∀ (a b : ↥(A ⊓ B)), Commute a.1 b.1)
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 :=
  H.rank_inf_eq_one_of_commute_of_flat_of_inj (Or.inl ‹_›) hc hinj


theorem rank_inf_eq_one_of_commute_of_flat_right_of_inj [Module.Flat R B]
    (hc : ∀ (a b : ↥(A ⊓ B)), Commute a.1 b.1)
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 :=
  H.rank_inf_eq_one_of_commute_of_flat_of_inj (Or.inr ‹_›) hc hinj


theorem rank_eq_one_of_commute_of_flat_of_self_of_inj (H : A.LinearDisjoint A) [Module.Flat R A]
    (hc : ∀ (a b : A), Commute a.1 b.1)
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R A = 1 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A : Subalgebra R S
    H : A.LinearDisjoint A
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem A x)
    hc : ∀ (a b : Subtype fun x => Membership.mem A x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem A x)) 1
  -/
  rw [← inf_of_le_left (le_refl A)] at hc ⊢
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    A : Subalgebra R S
    H : A.LinearDisjoint A
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem A x)
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A A) x), Commute ↑a ↑b
    hinj : Function.Injective ⇑(algebraMap R S)
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Min.min A A) x)) 1
  -/
  exact H.rank_inf_eq_one_of_commute_of_flat_left_of_inj hc hinj
  /-
    🎉 no goals
  -/


variable {A B} in
/-- In a commutative ring, if `A` and `B` are linearly disjoint, if `B` is a flat `R`-module,
then for any family of `R`-linearly independent elements of `A`,
they are also `B`-linearly independent. -/
theorem linearIndependent_left_of_flat (H : A.LinearDisjoint B) [Module.Flat R B]
    {ι : Type*} {a : ι → A} (ha : LinearIndependent R a) : LinearIndependent B (A.val ∘ a) :=
  H.linearIndependent_left_of_flat_of_commute ha fun _ _ ↦ mul_comm _ _


/-- In a commutative ring, if a basis of `A` is also `B`-linearly independent,
then `A` and `B` are linearly disjoint. -/
theorem of_basis_left {ι : Type*} (a : Basis ι R A)
    (H : LinearIndependent B (A.val ∘ a)) : A.LinearDisjoint B :=
  of_basis_left_of_commute A B a H fun _ _ ↦ mul_comm _ _


variable (R) in
/-- If `A` and `B` are flat algebras over `R`, such that `A ⊗[R] B` is a domain, and such that
the algebra maps are injective, then there exists an `R`-algebra `K` that is a field that `A`
and `B` inject into with linearly disjoint images. Note: `K` can chosen to be the
fraction field of `A ⊗[R] B`, but here we hide this fact. -/
theorem exists_field_of_isDomain_of_injective (A : Type v) [CommRing A] (B : Type w) [CommRing B]
    [Algebra R A] [Algebra R B] [Module.Flat R A] [Module.Flat R B] [IsDomain (A ⊗[R] B)]
    (ha : Function.Injective (algebraMap R A)) (hb : Function.Injective (algebraMap R B)) :
    ∃ (K : Type (max v w)) (_ : Field K) (_ : Algebra R K) (fa : A →ₐ[R] K) (fb : B →ₐ[R] K),
    Function.Injective fa ∧ Function.Injective fb ∧ fa.range.LinearDisjoint fb.range :=
  let K := FractionRing (A ⊗[R] B)
  let i := IsScalarTower.toAlgHom R (A ⊗[R] B) K
  have hi : Function.Injective i := IsFractionRing.injective (A ⊗[R] B) K
  ⟨K, inferInstance, inferInstance,
    i.comp Algebra.TensorProduct.includeLeft,
    i.comp Algebra.TensorProduct.includeRight,
    hi.comp (Algebra.TensorProduct.includeLeft_injective hb),
    hi.comp (Algebra.TensorProduct.includeRight_injective ha), by
      /-
        R : Type u
        inst✝⁷ : CommRing R
        A : Type v
        inst✝⁶ : CommRing A
        B : Type w
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : Module.Flat R A
        inst✝¹ : Module.Flat R B
        inst✝ : IsDomain (TensorProduct R A B)
        ha : Function.Injective ⇑(algebraMap R A)
        hb : Function.Injective ⇑(algebraMap R B)
        K : Type (max w v) := FractionRing (TensorProduct R A B)
        i : AlgHom R (TensorProduct R A B) K := IsScalarTower.toAlgHom R (TensorProduc …
        hi : Function.Injective ⇑i
        ⊢ (i.comp Algebra.TensorProduct.includeLeft).range.LinearDisjoint (i.comp Alge …
      -/
      simpa only [AlgHom.range_comp] using (include_range R A B).map i hi⟩
      /-
        🎉 no goals
      -/


/-- If `A ⊗[R] B` is a field, then `A` and `B` are linearly disjoint. -/
theorem of_isField (H : IsField (A ⊗[R] B)) : A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : IsField (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype fu …
    ⊢ A.LinearDisjoint B
  -/
  nontriviality S
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : IsField (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype fu …
    a✝ : Nontrivial S
    ⊢ A.LinearDisjoint B
  -/
  rw [linearDisjoint_iff_injective]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : IsField (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype fu …
    a✝ : Nontrivial S
    ⊢ Function.Injective ⇑(A.mulMap B)
  -/
  letI : Field (A ⊗[R] B) := H.toField
  -- need this otherwise `RingHom.injective` does not work
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : IsField (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype fu …
    a✝ : Nontrivial S
    this : Field (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype f …
    ⊢ Function.Injective ⇑(A.mulMap B)
  -/
  letI : NonAssocRing (A ⊗[R] B) := Ring.toNonAssocRing
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    H : IsField (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype fu …
    a✝ : Nontrivial S
    this✝ : Field (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype  …
    this : NonAssocRing (TensorProduct R (Subtype fun x => Membership.mem A x) (Su …
    ⊢ Function.Injective ⇑(A.mulMap B)
  -/
  exact RingHom.injective _
  /-
    🎉 no goals
  -/


/-- If `A ⊗[R] B` is a field, then for any `R`-algebra `S`
and injections of `A` and `B` into `S`, their images are linearly disjoint. -/
theorem of_isField' {A : Type v} [CommRing A] {B : Type w} [CommRing B]
    [Algebra R A] [Algebra R B] (H : IsField (A ⊗[R] B))
    (fa : A →ₐ[R] S) (fb : B →ₐ[R] S) (hfa : Function.Injective fa) (hfb : Function.Injective fb) :
    fa.range.LinearDisjoint fb.range := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A : Type v
    inst✝³ : CommRing A
    B : Type w
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    H : IsField (TensorProduct R A B)
    fa : AlgHom R A S
    fb : AlgHom R B S
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    ⊢ fa.range.LinearDisjoint fb.range
  -/
  apply of_isField
  exact Algebra.TensorProduct.congr (AlgEquiv.ofInjective fa hfa)
    (AlgEquiv.ofInjective fb hfb) |>.symm.toMulEquiv.isField _ H

-- need to be in this file since it uses linearly disjoint

open Cardinal Polynomial in
variable (R) in
/-- If `A` and `B` are flat `R`-algebras, both of them are transcendental, then `A ⊗[R] B` cannot
be a field. -/
theorem _root_.Algebra.TensorProduct.not_isField_of_transcendental
    (A : Type v) [CommRing A] (B : Type w) [CommRing B] [Algebra R A] [Algebra R B]
    [Module.Flat R A] [Module.Flat R B] [Algebra.Transcendental R A] [Algebra.Transcendental R B] :
    ¬IsField (A ⊗[R] B) := fun H ↦ by
  /-
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    ⊢ False
  -/
  letI := H.toField
  /-
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    ⊢ False
  -/
  obtain ⟨a, hta⟩ := ‹Algebra.Transcendental R A›
  /-
    case mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    ⊢ False
  -/
  obtain ⟨b, htb⟩ := ‹Algebra.Transcendental R B›
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ⊢ False
  -/
  have ha : Function.Injective (algebraMap R A) := Algebra.injective_of_transcendental
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    ⊢ False
  -/
  have hb : Function.Injective (algebraMap R B) := Algebra.injective_of_transcendental
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    ⊢ False
  -/
  let fa : A →ₐ[R] A ⊗[R] B := Algebra.TensorProduct.includeLeft
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    ⊢ False
  -/
  let fb : B →ₐ[R] A ⊗[R] B := Algebra.TensorProduct.includeRight
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    ⊢ False
  -/
  have hfa : Function.Injective fa := Algebra.TensorProduct.includeLeft_injective hb
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    ⊢ False
  -/
  have hfb : Function.Injective fb := Algebra.TensorProduct.includeRight_injective ha
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    ⊢ False
  -/
  haveI := hfa.isDomain fa.toRingHom
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this : IsDomain A
    ⊢ False
  -/
  haveI := hfb.isDomain fb.toRingHom
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝¹ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝ : IsDomain A
    this : IsDomain B
    ⊢ False
  -/
  haveI := ha.isDomain _
  haveI : Module.Flat R (toSubmodule fa.range) :=
    .of_linearEquiv _ _ _ (AlgEquiv.ofInjective fa hfa).symm.toLinearEquiv
  have key1 : Module.rank R ↥(fa.range ⊓ fb.range) ≤ 1 :=
    (include_range R A B).rank_inf_le_one_of_flat_left
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ⊢ False
  -/
  let ga : R[X] →ₐ[R] A := aeval a
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    ⊢ False
  -/
  let gb : R[X] →ₐ[R] B := aeval b
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    ⊢ False
  -/
  let gab := fa.comp ga
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    hta : Transcendental R a
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    ⊢ False
  -/
  replace hta : Function.Injective ga := transcendental_iff_injective.1 hta
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    htb : Transcendental R b
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    ⊢ False
  -/
  replace htb : Function.Injective gb := transcendental_iff_injective.1 htb
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    ⊢ False
  -/
  have htab : Function.Injective gab := hfa.comp hta
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    ⊢ False
  -/
  algebraize_only [ga.toRingHom, gb.toRingHom]
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    ⊢ False
  -/
  let f := Algebra.TensorProduct.mapOfCompatibleSMul R[X] R A B
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝³ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝² : IsDomain A
    this✝¹ : IsDomain B
    this✝ : IsDomain R
    this : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    f : AlgHom R (TensorProduct R A B) (TensorProduct (Polynomial R) A B) := Algeb …
    ⊢ False
  -/
  haveI := Algebra.TensorProduct.nontrivial_of_algebraMap_injective_of_isDomain R[X] A B hta htb
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝⁴ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝³ : IsDomain A
    this✝² : IsDomain B
    this✝¹ : IsDomain R
    this✝ : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    f : AlgHom R (TensorProduct R A B) (TensorProduct (Polynomial R) A B) := Algeb …
    this : Nontrivial (TensorProduct (Polynomial R) A B)
    ⊢ False
  -/
  have hf : Function.Injective f := RingHom.injective _
  have key2 : gab.range ≤ fa.range ⊓ fb.range := by
    simp_rw [gab, ga, ← aeval_algHom]
    rw [Algebra.TensorProduct.includeLeft_apply, ← Algebra.adjoin_singleton_eq_range_aeval]
    simp_rw [Algebra.adjoin_le_iff, Set.singleton_subset_iff, Algebra.coe_inf, Set.mem_inter_iff,
      AlgHom.coe_range, Set.mem_range]
    refine ⟨⟨a, by simp [fa]⟩, ⟨b, hf ?_⟩⟩
    simp_rw [fb, Algebra.TensorProduct.includeRight_apply, f,
      Algebra.TensorProduct.mapOfCompatibleSMul_tmul]
    convert ← (TensorProduct.smul_tmul (R := R[X]) (R' := R[X]) (M := A) (N := B) X 1 1).symm <;>
      (simp_rw [Algebra.smul_def, mul_one]; exact aeval_X _)
  have key3 := (Subalgebra.inclusion key2).comp (AlgEquiv.ofInjective gab htab).toAlgHom
    |>.toLinearMap.lift_rank_le_of_injective
      ((Subalgebra.inclusion_injective key2).comp (AlgEquiv.injective _))
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝⁴ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝³ : IsDomain A
    this✝² : IsDomain B
    this✝¹ : IsDomain R
    this✝ : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    f : AlgHom R (TensorProduct R A B) (TensorProduct (Polynomial R) A B) := Algeb …
    this : Nontrivial (TensorProduct (Polynomial R) A B)
    hf : Function.Injective ⇑f
    key2 : LE.le gab.range (Min.min fa.range fb.range)
    key3 : LE.le (Cardinal.lift.{max v w, u} (Module.rank R (Polynomial R))) (Card …
    ⊢ False
  -/
  have := lift_uzero.{u} _ ▸ (basisMonomials R).mk_eq_rank.symm
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝⁵ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝⁴ : IsDomain A
    this✝³ : IsDomain B
    this✝² : IsDomain R
    this✝¹ : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodul …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    f : AlgHom R (TensorProduct R A B) (TensorProduct (Polynomial R) A B) := Algeb …
    this✝ : Nontrivial (TensorProduct (Polynomial R) A B)
    hf : Function.Injective ⇑f
    key2 : LE.le gab.range (Min.min fa.range fb.range)
    key3 : LE.le (Cardinal.lift.{max v w, u} (Module.rank R (Polynomial R))) (Card …
    this : Eq (Module.rank R (Polynomial R)) (Cardinal.lift.{u, 0} (Cardinal.mk Na …
    ⊢ False
  -/
  simp only [this, mk_eq_aleph0, lift_aleph0, aleph0_le_lift] at key3
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝⁸ : CommRing R
    A : Type v
    inst✝⁷ : CommRing A
    B : Type w
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Module.Flat R A
    inst✝² : Module.Flat R B
    inst✝¹ : Algebra.Transcendental R A
    inst✝ : Algebra.Transcendental R B
    H : IsField (TensorProduct R A B)
    this✝⁵ : Field (TensorProduct R A B) := H.toField
    a : A
    b : B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    fa : AlgHom R A (TensorProduct R A B) := Algebra.TensorProduct.includeLeft
    fb : AlgHom R B (TensorProduct R A B) := Algebra.TensorProduct.includeRight
    hfa : Function.Injective ⇑fa
    hfb : Function.Injective ⇑fb
    this✝⁴ : IsDomain A
    this✝³ : IsDomain B
    this✝² : IsDomain R
    this✝¹ : Module.Flat R (Subtype fun x => Membership.mem (Subalgebra.toSubmodul …
    key1 : LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min fa.range …
    ga : AlgHom R (Polynomial R) A := Polynomial.aeval a
    gb : AlgHom R (Polynomial R) B := Polynomial.aeval b
    gab : AlgHom R (Polynomial R) (TensorProduct R A B) := fa.comp ga
    hta : Function.Injective ⇑ga
    htb : Function.Injective ⇑gb
    htab : Function.Injective ⇑gab
    algInst✝¹ : Algebra (Polynomial R) A := ga.toAlgebra
    algInst✝ : Algebra (Polynomial R) B := gb.toAlgebra
    f : AlgHom R (TensorProduct R A B) (TensorProduct (Polynomial R) A B) := Algeb …
    this✝ : Nontrivial (TensorProduct (Polynomial R) A B)
    hf : Function.Injective ⇑f
    key2 : LE.le gab.range (Min.min fa.range fb.range)
    this : Eq (Module.rank R (Polynomial R)) (Cardinal.lift.{u, 0} (Cardinal.mk Na …
    key3 : LE.le Cardinal.aleph0 (Module.rank R (Subtype fun x => Membership.mem ( …
    ⊢ False
  -/
  exact (key3.trans key1).not_lt one_lt_aleph0
  /-
    🎉 no goals
  -/


variable (R) in
/-- If `A` and `B` are flat `R`-algebras, such that `A ⊗[R] B` is a field, then one of `A` and `B`
is algebraic over `R`. -/
theorem _root_.Algebra.TensorProduct.isAlgebraic_of_isField
    (A : Type v) [CommRing A] (B : Type w) [CommRing B] [Algebra R A] [Algebra R B]
    [Module.Flat R A] [Module.Flat R B] (H : IsField (A ⊗[R] B)) :
    Algebra.IsAlgebraic R A ∨ Algebra.IsAlgebraic R B := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type v
    inst✝⁵ : CommRing A
    B : Type w
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Module.Flat R A
    inst✝ : Module.Flat R B
    H : IsField (TensorProduct R A B)
    ⊢ Or (Algebra.IsAlgebraic R A) (Algebra.IsAlgebraic R B)
  -/
  by_contra! h
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type v
    inst✝⁵ : CommRing A
    B : Type w
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Module.Flat R A
    inst✝ : Module.Flat R B
    H : IsField (TensorProduct R A B)
    h : And (Not (Algebra.IsAlgebraic R A)) (Not (Algebra.IsAlgebraic R B))
    ⊢ False
  -/
  simp_rw [← Algebra.transcendental_iff_not_isAlgebraic] at h
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type v
    inst✝⁵ : CommRing A
    B : Type w
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Module.Flat R A
    inst✝ : Module.Flat R B
    H : IsField (TensorProduct R A B)
    h : And (Algebra.Transcendental R A) (Algebra.Transcendental R B)
    ⊢ False
  -/
  obtain ⟨_, _⟩ := h
  /-
    case intro
    R : Type u
    inst✝⁶ : CommRing R
    A : Type v
    inst✝⁵ : CommRing A
    B : Type w
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Module.Flat R A
    inst✝ : Module.Flat R B
    H : IsField (TensorProduct R A B)
    left✝ : Algebra.Transcendental R A
    right✝ : Algebra.Transcendental R B
    ⊢ False
  -/
  exact Algebra.TensorProduct.not_isField_of_transcendental R A B H
  /-
    🎉 no goals
  -/


include H in
theorem rank_inf_eq_one_of_flat_of_inj (hf : Module.Flat R A ∨ Module.Flat R B)
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 :=
  H.rank_inf_eq_one_of_commute_of_flat_of_inj hf (fun _ _ ↦ mul_comm _ _) hinj


include H in
theorem rank_inf_eq_one_of_flat_left_of_inj [Module.Flat R A]
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 :=
  H.rank_inf_eq_one_of_commute_of_flat_left_of_inj (fun _ _ ↦ mul_comm _ _) hinj


include H in
theorem rank_inf_eq_one_of_flat_right_of_inj [Module.Flat R B]
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R ↥(A ⊓ B) = 1 :=
  H.rank_inf_eq_one_of_commute_of_flat_right_of_inj (fun _ _ ↦ mul_comm _ _) hinj


theorem rank_eq_one_of_flat_of_self_of_inj (H : A.LinearDisjoint A) [Module.Flat R A]
    (hinj : Function.Injective (algebraMap R S)) : Module.rank R A = 1 :=
  H.rank_eq_one_of_commute_of_flat_of_self_of_inj (fun _ _ ↦ mul_comm _ _) hinj


include H in
/-- In a commutative ring, if subalgebras `A` and `B` are linearly disjoint and they are
free modules, then the rank of `A ⊔ B` is equal to the product of the rank of `A` and `B`. -/
theorem rank_sup_of_free [Module.Free R A] [Module.Free R B] :
    Module.rank R ↥(A ⊔ B) = Module.rank R A * Module.rank R B := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free R (Subtype fun x => Membership.mem B x)
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free R (Subtype fun x => Membership.mem B x)
    a✝ : Nontrivial R
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMul.h …
  -/
  rw [← rank_tensorProduct', H.mulMap.toLinearEquiv.rank_eq]
  /-
    🎉 no goals
  -/


include H in
/-- In a commutative ring, if subalgebras `A` and `B` are linearly disjoint and they are
free modules, then the rank of `A ⊔ B` is equal to the product of the rank of `A` and `B`. -/
theorem finrank_sup_of_free [Module.Free R A] [Module.Free R B] :
    Module.finrank R ↥(A ⊔ B) = Module.finrank R A * Module.finrank R B := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Free R (Subtype fun x => Membership.mem B x)
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (HMu …
  -/
  simpa only [map_mul] using congr(Cardinal.toNat $(H.rank_sup_of_free))
  /-
    🎉 no goals
  -/


/-- In a commutative ring, if `A` and `B` are subalgebras which are free modules of finite rank,
such that rank of `A ⊔ B` is equal to the product of the rank of `A` and `B`,
then `A` and `B` are linearly disjoint. -/
theorem of_finrank_sup_of_free [Module.Free R A] [Module.Free R B]
    [Module.Finite R A] [Module.Finite R B]
    (H : Module.finrank R ↥(A ⊔ B) = Module.finrank R A * Module.finrank R B) :
    A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (H …
    ⊢ A.LinearDisjoint B
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (H …
    a✝ : Nontrivial R
    ⊢ A.LinearDisjoint B
  -/
  rw [← Module.finrank_tensorProduct] at H
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    ⊢ A.LinearDisjoint B
  -/
  obtain ⟨j, hj⟩ := exists_linearIndependent_of_le_finrank H.ge
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    hj : LinearIndependent R j
    ⊢ A.LinearDisjoint B
  -/
  rw [LinearIndependent] at hj
  let j' := Finsupp.linearCombination R j ∘ₗ
    (LinearEquiv.ofFinrankEq (A ⊗[R] B) _ (by simp)).toLinearMap
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    hj : Function.Injective ⇑(Finsupp.linearCombination R j)
    j' : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    ⊢ A.LinearDisjoint B
  -/
  replace hj : Function.Injective j' := by simpa [j']
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    j' : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    hj : Function.Injective ⇑j'
    ⊢ A.LinearDisjoint B
  -/
  have hf : Function.Surjective (mulMap' A B).toLinearMap := mulMap'_surjective A B
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    j' : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    hj : Function.Injective ⇑j'
    hf : Function.Surjective ⇑(A.mulMap' B).toLinearMap
    ⊢ A.LinearDisjoint B
  -/
  haveI := Subalgebra.finite_sup A B
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    j' : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    hj : Function.Injective ⇑j'
    hf : Function.Surjective ⇑(A.mulMap' B).toLinearMap
    this : Module.Finite R (Subtype fun x => Membership.mem (Max.max A B) x)
    ⊢ A.LinearDisjoint B
  -/
  rw [linearDisjoint_iff, Submodule.linearDisjoint_iff]
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Finite R (Subtype fun x => Membership.mem A x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem B x)
    H : Eq (Module.finrank R (Subtype fun x => Membership.mem (Max.max A B) x)) (M …
    a✝ : Nontrivial R
    j : Fin (Module.finrank R (TensorProduct R (Subtype fun x => Membership.mem A  …
    j' : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    hj : Function.Injective ⇑j'
    hf : Function.Surjective ⇑(A.mulMap' B).toLinearMap
    this : Module.Finite R (Subtype fun x => Membership.mem (Max.max A B) x)
    ⊢ Function.Injective ⇑((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodu …
  -/
  exact Subtype.val_injective.comp (OrzechProperty.injective_of_surjective_of_injective j' _ hj hf)
  /-
    🎉 no goals
  -/


include H in
/-- If `A` and `B` are linearly disjoint, if `A` is free and `B` is flat,
then `[B[A] : B] = [A : R]`. See also `Subalgebra.adjoin_rank_le`. -/
theorem adjoin_rank_eq_rank_left [Module.Free R A] [Module.Flat R B]
    [Nontrivial R] [Nontrivial S] :
    Module.rank B (Algebra.adjoin B (A : Set S)) = Module.rank R A := by
  rw [← rank_toSubmodule, Module.Free.rank_eq_card_chooseBasisIndex R A,
    A.adjoin_eq_span_basis B (Module.Free.chooseBasis R A)]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Flat R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Nontrivial R
    inst✝ : Nontrivial S
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem B x) (Subtype fun x => Memb …
  -/
  change Module.rank B (Submodule.span B (Set.range (A.val ∘ Module.Free.chooseBasis R A))) = _
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Flat R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Nontrivial R
    inst✝ : Nontrivial S
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem B x) (Subtype fun x => Memb …
  -/
  have := H.linearIndependent_left_of_flat (Module.Free.chooseBasis R A).linearIndependent
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    H : A.LinearDisjoint B
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Flat R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Nontrivial R
    inst✝ : Nontrivial S
    this : LinearIndependent (Subtype fun x => Membership.mem B x) (Function.comp  …
    ⊢ Eq (Module.rank (Subtype fun x => Membership.mem B x) (Subtype fun x => Memb …
  -/
  rw [rank_span this, Cardinal.mk_range_eq _ this.injective]
  /-
    🎉 no goals
  -/


include H in
/-- If `A` and `B` are linearly disjoint, if `B` is free and `A` is flat,
then `[A[B] : A] = [B : R]`. See also `Subalgebra.adjoin_rank_le`. -/
theorem adjoin_rank_eq_rank_right [Module.Free R B] [Module.Flat R A]
    [Nontrivial R] [Nontrivial S] :
    Module.rank A (Algebra.adjoin A (B : Set S)) = Module.rank R B :=
  H.symm.adjoin_rank_eq_rank_left


/-- If the rank of `A` and `B` are coprime, and they satisfy some freeness condition,
then `A` and `B` are linearly disjoint. -/
theorem of_finrank_coprime_of_free [Module.Free R A] [Module.Free R B]
    [Module.Free A (Algebra.adjoin A (B : Set S))] [Module.Free B (Algebra.adjoin B (A : Set S))]
    (H : (Module.finrank R A).Coprime (Module.finrank R B)) : A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    ⊢ A.LinearDisjoint B
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    a✝ : Nontrivial R
    ⊢ A.LinearDisjoint B
  -/
  by_cases h1 : Module.finrank R A = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
      a✝ : Nontrivial R
      h1 : Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0
      ⊢ A.LinearDisjoint B
    -/
  · rw [h1, Nat.coprime_zero_left] at H
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 1
      a✝ : Nontrivial R
      h1 : Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0
      ⊢ A.LinearDisjoint B
    -/
    rw [eq_bot_of_finrank_one H]
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 1
      a✝ : Nontrivial R
      h1 : Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0
      ⊢ A.LinearDisjoint Bot.bot
    -/
    exact bot_right _
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    a✝ : Nontrivial R
    h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
    ⊢ A.LinearDisjoint B
  -/
  by_cases h2 : Module.finrank R B = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
      a✝ : Nontrivial R
      h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
      h2 : Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0
      ⊢ A.LinearDisjoint B
    -/
  · rw [h2, Nat.coprime_zero_right] at H
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 1
      a✝ : Nontrivial R
      h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
      h2 : Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0
      ⊢ A.LinearDisjoint B
    -/
    rw [eq_bot_of_finrank_one H]
    /-
      case pos
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      A B : Subalgebra R S
      inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
      inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
      inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
      inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
      H : Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 1
      a✝ : Nontrivial R
      h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
      h2 : Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0
      ⊢ Bot.bot.LinearDisjoint B
    -/
    exact bot_left _
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    a✝ : Nontrivial R
    h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
    h2 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0)
    ⊢ A.LinearDisjoint B
  -/
  haveI := Module.finite_of_finrank_pos (Nat.pos_of_ne_zero h1)
  /-
    case neg
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    a✝ : Nontrivial R
    h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
    h2 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0)
    this : Module.Finite R (Subtype fun x => Membership.mem A x)
    ⊢ A.LinearDisjoint B
  -/
  haveI := Module.finite_of_finrank_pos (Nat.pos_of_ne_zero h2)
  /-
    case neg
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    A B : Subalgebra R S
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem A x)
    inst✝² : Module.Free R (Subtype fun x => Membership.mem B x)
    inst✝¹ : Module.Free (Subtype fun x => Membership.mem A x) (Subtype fun x => M …
    inst✝ : Module.Free (Subtype fun x => Membership.mem B x) (Subtype fun x => Me …
    H : (Module.finrank R (Subtype fun x => Membership.mem A x)).Coprime (Module.f …
    a✝ : Nontrivial R
    h1 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem A x)) 0)
    h2 : Not (Eq (Module.finrank R (Subtype fun x => Membership.mem B x)) 0)
    this✝ : Module.Finite R (Subtype fun x => Membership.mem A x)
    this : Module.Finite R (Subtype fun x => Membership.mem B x)
    ⊢ A.LinearDisjoint B
  -/
  haveI := finite_sup A B
  have : Module.finrank R A ≤ Module.finrank R ↥(A ⊔ B) :=
    LinearMap.finrank_le_finrank_of_injective <|
      Submodule.inclusion_injective (show toSubmodule A ≤ toSubmodule (A ⊔ B) by simp)
  exact of_finrank_sup_of_free <| (finrank_sup_le_of_free A B).antisymm <|
    Nat.le_of_dvd (lt_of_lt_of_le (Nat.pos_of_ne_zero h1) this) <| H.mul_dvd_of_dvd_of_dvd
      (finrank_left_dvd_finrank_sup_of_free A B) (finrank_right_dvd_finrank_sup_of_free A B)


/-- If `A/R` is integral, such that `A'` and `B` are linearly disjoint for all subalgebras `A'`
of `A` which are finitely generated `R`-modules, then `A` and `B` are linearly disjoint. -/
theorem of_linearDisjoint_finite_left [Algebra.IsIntegral R A]
    (H : ∀ A' : Subalgebra R S, A' ≤ A → [Module.Finite R A'] → A'.LinearDisjoint B) :
    A.LinearDisjoint B := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    ⊢ A.LinearDisjoint B
  -/
  rw [linearDisjoint_iff, Submodule.linearDisjoint_iff]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    ⊢ Function.Injective ⇑((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodu …
  -/
  intro x y hxy
  obtain ⟨M', hM, hf, h⟩ :=
    TensorProduct.exists_finite_submodule_left_of_finite' {x, y} (Set.toFinite _)
  /-
    case intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  obtain ⟨s, hs⟩ := Module.Finite.iff_fg.1 hf
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    ⊢ Eq x y
  -/
  have hs' : (s : Set S) ⊆ A := by rwa [← hs, Submodule.span_le] at hM
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    ⊢ Eq x y
  -/
  let A' := Algebra.adjoin R (s : Set S)
  have hf' : Submodule.FG (toSubmodule A') := fg_adjoin_of_finite s.finite_toSet fun x hx ↦
    (isIntegral_algHom_iff A.val Subtype.val_injective).2
      (Algebra.IsIntegral.isIntegral (R := R) (A := A) ⟨x, hs' hx⟩)
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : (Subalgebra.toSubmodule A').FG
    ⊢ Eq x y
  -/
  replace hf' : Module.Finite R A' := Module.Finite.iff_fg.2 hf'
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : Module.Finite R (Subtype fun x => Membership.mem A' x)
    ⊢ Eq x y
  -/
  have hA : toSubmodule A' ≤ toSubmodule A := Algebra.adjoin_le_iff.2 hs'
  replace h : {x, y} ⊆ (LinearMap.range (LinearMap.rTensor (toSubmodule B)
      (Submodule.inclusion hA)) : Set _) := fun _ hx ↦ by
    have : Submodule.inclusion hM = Submodule.inclusion hA ∘ₗ Submodule.inclusion
      (show M' ≤ toSubmodule A' by
        rw [← hs, Submodule.span_le]; exact Algebra.adjoin_le_iff.1 (le_refl _)) := rfl
    rw [this, LinearMap.rTensor_comp] at h
    exact LinearMap.range_comp_le_range _ _ (h hx)
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : Module.Finite R (Subtype fun x => Membership.mem A' x)
    hA : LE.le (Subalgebra.toSubmodule A') (Subalgebra.toSubmodule A)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  obtain ⟨x', hx'⟩ := h (show x ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : Module.Finite R (Subtype fun x => Membership.mem A' x)
    hA : LE.le (Subalgebra.toSubmodule A') (Subalgebra.toSubmodule A)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem (Subalgebra.toSu …
    ⊢ Eq x y
  -/
  obtain ⟨y', hy'⟩ := h (show y ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : Module.Finite R (Subtype fun x => Membership.mem A' x)
    hA : LE.le (Subalgebra.toSubmodule A') (Subalgebra.toSubmodule A)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem (Subalgebra.toSu …
    y' : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    hy' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem (Subalgebra.toSu …
    ⊢ Eq x y
  -/
  rw [← hx', ← hy']; congr
  /-
    case intro.intro.intro.intro.intro.intro.h.e_6.h
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    A B : Subalgebra R S
    inst✝ : Algebra.IsIntegral R (Subtype fun x => Membership.mem A x)
    H : ∀ (A' : Subalgebra R S), LE.le A' A → ∀ [inst : Module.Finite R (Subtype f …
    x y : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule …
    hxy : Eq (((Subalgebra.toSubmodule A).mulMap (Subalgebra.toSubmodule B)) x) (( …
    M' : Submodule R S
    hM : LE.le M' (Subalgebra.toSubmodule A)
    hf : Module.Finite R (Subtype fun x => Membership.mem M' x)
    s : Finset S
    hs : Eq (Submodule.span R ↑s) M'
    hs' : HasSubset.Subset ↑s ↑A
    A' : Subalgebra R S := Algebra.adjoin R ↑s
    hf' : Module.Finite R (Subtype fun x => Membership.mem A' x)
    hA : LE.le (Subalgebra.toSubmodule A') (Subalgebra.toSubmodule A)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem (Subalgebra.toSu …
    y' : TensorProduct R (Subtype fun x => Membership.mem (Subalgebra.toSubmodule  …
    hy' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem (Subalgebra.toSu …
    ⊢ Eq x' y'
  -/
  exact (H A' hA).injective (by simp [← Submodule.mulMap_comp_rTensor _ hA, hx', hy', hxy])
  /-
    🎉 no goals
  -/


/-- If `B/R` is integral, such that `A` and `B'` are linearly disjoint for all subalgebras `B'`
of `B` which are finitely generated `R`-modules, then `A` and `B` are linearly disjoint. -/
theorem of_linearDisjoint_finite_right [Algebra.IsIntegral R B]
    (H : ∀ B' : Subalgebra R S, B' ≤ B → [Module.Finite R B'] → A.LinearDisjoint B') :
    A.LinearDisjoint B :=
  (of_linearDisjoint_finite_left B A fun B' hB' _ ↦ (H B' hB').symm).symm


/-- If `A/R` and `B/R` are integral, such that any finite subalgebras in `A` and `B` are
linearly disjoint, then `A` and `B` are linearly disjoint. -/
theorem of_linearDisjoint_finite
    [Algebra.IsIntegral R A] [Algebra.IsIntegral R B]
    (H : ∀ (A' B' : Subalgebra R S), A' ≤ A → B' ≤ B →
      [Module.Finite R A'] → [Module.Finite R B'] → A'.LinearDisjoint B') :
    A.LinearDisjoint B :=
  of_linearDisjoint_finite_left A B fun _ hA' _ ↦
    of_linearDisjoint_finite_right _ B fun _ hB' _ ↦ H _ _ hA' hB'


theorem inf_eq_bot_of_commute (H : A.LinearDisjoint B)
    (hc : ∀ (a b : ↥(A ⊓ B)), Commute a.1 b.1) : A ⊓ B = ⊥ :=
  eq_bot_of_rank_le_one (Submodule.LinearDisjoint.rank_inf_le_one_of_commute_of_flat_left H hc)


theorem eq_bot_of_commute_of_self (H : A.LinearDisjoint A)
    (hc : ∀ (a b : A), Commute a.1 b.1) : A = ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝² : Field R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A : Subalgebra R S
    H : A.LinearDisjoint A
    hc : ∀ (a b : Subtype fun x => Membership.mem A x), Commute ↑a ↑b
    ⊢ Eq A Bot.bot
  -/
  rw [← inf_of_le_left (le_refl A)] at hc ⊢
  /-
    R : Type u
    S : Type v
    inst✝² : Field R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    A : Subalgebra R S
    H : A.LinearDisjoint A
    hc : ∀ (a b : Subtype fun x => Membership.mem (Min.min A A) x), Commute ↑a ↑b
    ⊢ Eq (Min.min A A) Bot.bot
  -/
  exact H.inf_eq_bot_of_commute hc
  /-
    🎉 no goals
  -/


theorem inf_eq_bot (H : A.LinearDisjoint B) : A ⊓ B = ⊥ :=
  H.inf_eq_bot_of_commute fun _ _ ↦ mul_comm _ _


theorem eq_bot_of_self (H : A.LinearDisjoint A) : A = ⊥ :=
  H.eq_bot_of_commute_of_self fun _ _ ↦ mul_comm _ _



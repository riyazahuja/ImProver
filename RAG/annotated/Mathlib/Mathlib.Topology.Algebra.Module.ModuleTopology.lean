/-- The module topology, for a module `A` over a topological ring `R`. It's the finest topology
making addition and the `R`-action continuous, or equivalently the finest topology making `A`
into a topological `R`-module. More precisely it's the Inf of the set of
topologies with these properties; theorems `continuousSMul` and `continuousAdd` show
that the module topology also has these properties. -/
abbrev moduleTopology : TopologicalSpace A :=
  sInf {t | @ContinuousSMul R A _ _ t ∧ @ContinuousAdd A t _}


/-- A class asserting that the topology on a module over a topological ring `R` is
the module topology. See `moduleTopology` for more discussion of the module topology. -/
class IsModuleTopology [τA : TopologicalSpace A] : Prop where
  /-- Note that this should not be used directly, and `eq_moduleTopology`, which takes `R` and `A`
  explicitly, should be used instead. -/
  eq_moduleTopology' : τA = moduleTopology R A


theorem eq_moduleTopology [τA : TopologicalSpace A] [IsModuleTopology R A] :
    τA = moduleTopology R A :=
  IsModuleTopology.eq_moduleTopology' (R := R) (A := A)


/-- Scalar multiplication `• : R × A → A` is continuous if `R` is a topological
ring, and `A` is an `R` module with the module topology. -/
theorem ModuleTopology.continuousSMul : @ContinuousSMul R A _ _ (moduleTopology R A) :=
  /- Proof: We need to prove that the product topology is finer than the pullback
     of the module topology. But the module topology is an Inf and thus a limit,
     and pullback is a right adjoint, so it preserves limits.
     We must thus show that the product topology is finer than an Inf, so it suffices
     to show it's a lower bound, which is not hard. All this is wrapped into
     `continuousSMul_sInf`.
  -/
  continuousSMul_sInf fun _ h ↦ h.1


/-- Addition `+ : A × A → A` is continuous if `R` is a topological
ring, and `A` is an `R` module with the module topology. -/
theorem ModuleTopology.continuousAdd : @ContinuousAdd A (moduleTopology R A) _ :=
  continuousAdd_sInf fun _ h ↦ h.2


instance IsModuleTopology.toContinuousSMul [TopologicalSpace A] [IsModuleTopology R A] :
    ContinuousSMul R A := eq_moduleTopology R A ▸ ModuleTopology.continuousSMul R A

-- this can't be an instance because typclass inference can't be expected to find `R`.

theorem IsModuleTopology.toContinuousAdd [TopologicalSpace A] [IsModuleTopology R A] :
    ContinuousAdd A := eq_moduleTopology R A ▸ ModuleTopology.continuousAdd R A


/-- The module topology is `≤` any topology making `A` into a topological module. -/
theorem moduleTopology_le [τA : TopologicalSpace A] [ContinuousSMul R A] [ContinuousAdd A] :
    moduleTopology R A ≤ τA := sInf_le ⟨inferInstance, inferInstance⟩


/-- If `A` is a topological `R`-module and the identity map from (`A` with its given
topology) to (`A` with the module topology) is continuous, then the topology on `A` is
the module topology. -/
theorem of_continuous_id [ContinuousAdd A] [ContinuousSMul R A]
    (h : @Continuous A A τA (moduleTopology R A) id):
    IsModuleTopology R A where
  -- The topologies are equal because each is finer than the other. One inclusion
  -- follows from the continuity hypothesis; the other is because the module topology
  -- is the inf of all the topologies making `A` a topological module.
  eq_moduleTopology' := le_antisymm (continuous_id_iff_le.1 h) (moduleTopology_le _ _)


/-- The zero module has the module topology. -/
instance instSubsingleton [Subsingleton A] : IsModuleTopology R A where
  eq_moduleTopology' := Subsingleton.elim _ _


/-- If `A` and `B` are `R`-modules, homeomorphic via an `R`-linear homeomorphism, and if
`A` has the module topology, then so does `B`. -/
theorem iso (e : A ≃L[R] B) : IsModuleTopology R B where
  eq_moduleTopology' := by
    -- get these in before I start putting new topologies on A and B and have to use `@`
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      ⊢ Eq τB (moduleTopology R B)
    -/
    let g : A →ₗ[R] B := e
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      ⊢ Eq τB (moduleTopology R B)
    -/
    let g' : B →ₗ[R] A := e.symm
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      ⊢ Eq τB (moduleTopology R B)
    -/
    let h : A →+ B := e
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      ⊢ Eq τB (moduleTopology R B)
    -/
    let h' : B →+ A := e.symm
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      h' : AddMonoidHom B A := ↑e.symm
      ⊢ Eq τB (moduleTopology R B)
    -/
    simp_rw [e.toHomeomorph.symm.isInducing.1, eq_moduleTopology R A, moduleTopology, induced_sInf]
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      h' : AddMonoidHom B A := ↑e.symm
      ⊢ Eq (InfSet.sInf (Set.image (TopologicalSpace.induced ⇑e.toHomeomorph.symm) ( …
    -/
    apply congr_arg
    /-
      case h
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      h' : AddMonoidHom B A := ↑e.symm
      ⊢ Eq (Set.image (TopologicalSpace.induced ⇑e.toHomeomorph.symm) (setOf fun t = …
    -/
    ext τ -- from this point on the definitions of `g`, `g'` etc above don't work without `@`.
    /-
      case h.h
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      h' : AddMonoidHom B A := ↑e.symm
      τ : TopologicalSpace B
      ⊢ Iff (Membership.mem (Set.image (TopologicalSpace.induced ⇑e.toHomeomorph.sym …
    -/
    rw [Set.mem_image]
    /-
      case h.h
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁵ : Semiring R
      A : Type u_2
      inst✝⁴ : AddCommMonoid A
      inst✝³ : Module R A
      τA : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommMonoid B
      inst✝ : Module R B
      τB : TopologicalSpace B
      e : ContinuousLinearEquiv (RingHom.id R) A B
      g : LinearMap (RingHom.id R) A B := ↑↑e
      g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
      h : AddMonoidHom A B := ↑e
      h' : AddMonoidHom B A := ↑e.symm
      τ : TopologicalSpace B
      ⊢ Iff (Exists fun x => And (Membership.mem (setOf fun t => And (ContinuousSMul …
    -/
    constructor
      /-
        case h.h.mp
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        ⊢ (Exists fun x => And (Membership.mem (setOf fun t => And (ContinuousSMul R A …
      -/
    · rintro ⟨σ, ⟨hσ1, hσ2⟩, rfl⟩
      /-
        case h.h.mp.intro.intro.intro
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        σ : TopologicalSpace A
        hσ1 : ContinuousSMul R A
        hσ2 : ContinuousAdd A
        ⊢ Membership.mem (setOf fun t => And (ContinuousSMul R B) (ContinuousAdd B)) ( …
      -/
      exact ⟨continuousSMul_induced g'.toMulActionHom, continuousAdd_induced h'⟩
      /-
        🎉 no goals
      -/
      /-
        case h.h.mpr
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        ⊢ Membership.mem (setOf fun t => And (ContinuousSMul R B) (ContinuousAdd B)) τ …
      -/
    · rintro ⟨h1, h2⟩
      /-
        case h.h.mpr.intro
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        h1 : ContinuousSMul R B
        h2 : ContinuousAdd B
        ⊢ Exists fun x => And (Membership.mem (setOf fun t => And (ContinuousSMul R A) …
      -/
      use τ.induced e
      /-
        case h
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        h1 : ContinuousSMul R B
        h2 : ContinuousAdd B
        ⊢ And (Membership.mem (setOf fun t => And (ContinuousSMul R A) (ContinuousAdd  …
      -/
      rw [induced_compose]
      /-
        case h
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        h1 : ContinuousSMul R B
        h2 : ContinuousAdd B
        ⊢ And (Membership.mem (setOf fun t => And (ContinuousSMul R A) (ContinuousAdd  …
      -/
      refine ⟨⟨continuousSMul_induced g.toMulActionHom, continuousAdd_induced h⟩, ?_⟩
      /-
        case h
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        h1 : ContinuousSMul R B
        h2 : ContinuousAdd B
        ⊢ Eq (TopologicalSpace.induced (Function.comp ⇑e ⇑e.toHomeomorph.symm) τ) τ
      -/
      nth_rw 2 [← induced_id (t := τ)]
      /-
        case h
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁵ : Semiring R
        A : Type u_2
        inst✝⁴ : AddCommMonoid A
        inst✝³ : Module R A
        τA : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommMonoid B
        inst✝ : Module R B
        τB : TopologicalSpace B
        e : ContinuousLinearEquiv (RingHom.id R) A B
        g : LinearMap (RingHom.id R) A B := ↑↑e
        g' : LinearMap (RingHom.id R) B A := ↑↑e.symm
        h : AddMonoidHom A B := ↑e
        h' : AddMonoidHom B A := ↑e.symm
        τ : TopologicalSpace B
        h1 : ContinuousSMul R B
        h2 : ContinuousAdd B
        ⊢ Eq (TopologicalSpace.induced (Function.comp ⇑e ⇑e.toHomeomorph.symm) τ) (Top …
      -/
      simp
      /-
        🎉 no goals
      -/


/-- The topology on a topological semiring `R` agrees with the module topology when considering
`R` as an `R`-module in the obvious way (i.e., via `Semiring.toModule`). -/
instance _root_.TopologicalSemiring.toIsModuleTopology : IsModuleTopology R R := by
  /- By a previous lemma it suffices to show that the identity from (R,usual) to
  (R, module topology) is continuous. -/
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    τR : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    ⊢ IsModuleTopology R R
  -/
  apply of_continuous_id
  /-
  The idea needed here is to rewrite the identity function as the composite of `r ↦ (r,1)`
  from `R` to `R × R`, and multiplication `R × R → R`.
  -/
  /-
    case h
    R : Type u_1
    inst✝¹ : Semiring R
    τR : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    ⊢ Continuous id
  -/
  rw [show (id : R → R) = (fun rs ↦ rs.1 • rs.2) ∘ (fun r ↦ (r, 1)) by ext; simp]
  /-
  It thus suffices to show that each of these maps are continuous. For this claim to even make
  sense, we need to topologise `R × R`. The trick is to do this by giving the first `R` the usual
  topology `τR` and the second `R` the module topology. To do this we have to "fight mathlib"
  a bit with `@`, because there is more than one topology on `R` here.
  -/
  apply @Continuous.comp R (R × R) R τR (@instTopologicalSpaceProd R R τR (moduleTopology R R))
      (moduleTopology R R)
  · /-
    The map R × R → R is `•`, so by a fundamental property of the module topology,
    this is continuous. -/
    /-
      case h.hg
      R : Type u_1
      inst✝¹ : Semiring R
      τR : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      ⊢ Continuous fun rs => HSMul.hSMul rs.1 rs.2
    -/
    exact @continuous_smul _ _ _ _ (moduleTopology R R) <| ModuleTopology.continuousSMul ..
    /-
      🎉 no goals
    -/
  · /-
    The map `R → R × R` sending `r` to `(r,1)` is a map into a product, so it suffices to show
    that each of the two factors is continuous. But the first is the identity function
    on `(R, usual topology)` and the second is a constant function. -/
    exact @Continuous.prod_mk _ _ _ _ (moduleTopology R R) _ _ _ continuous_id <|
      @continuous_const _ _ _ (moduleTopology R R) _


/-- The module topology coming from the action of the topological ring `Rᵐᵒᵖ` on `R`
  (via `Semiring.toOppositeModule`, i.e. via `(op r) • m = m * r`) is `R`'s topology. -/
instance _root_.TopologicalSemiring.toOppositeIsModuleTopology : IsModuleTopology Rᵐᵒᵖ R :=
  .iso (MulOpposite.opContinuousLinearEquiv Rᵐᵒᵖ).symm


/-- Every `R`-linear map between two topological `R`-modules, where the source has the module
topology, is continuous. -/
@[fun_prop, continuity]
theorem continuous_of_distribMulActionHom (φ : A →+[R] B) : Continuous φ := by
  -- the proof: We know that `+ : B × B → B` and `• : R × B → B` are continuous for the module
  -- topology on `B`, and two earlier theorems (`continuousSMul_induced` and
  -- `continuousAdd_induced`) say that hence `+` and `•` on `A` are continuous if `A`
  -- is given the topology induced from `φ`. Hence the module topology is finer than
  -- the induced topology, and so the function is continuous.
  /-
    R : Type u_1
    τR : TopologicalSpace R
    inst✝⁷ : Semiring R
    A : Type u_2
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    aA : TopologicalSpace A
    inst✝⁴ : IsModuleTopology R A
    B : Type u_3
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    aB : TopologicalSpace B
    inst✝¹ : ContinuousAdd B
    inst✝ : ContinuousSMul R B
    φ : DistribMulActionHom (MonoidHom.id R) A B
    ⊢ Continuous ⇑φ
  -/
  rw [eq_moduleTopology R A, continuous_iff_le_induced]
  exact sInf_le <| ⟨continuousSMul_induced (φ.toMulActionHom),
    continuousAdd_induced φ.toAddMonoidHom⟩


@[fun_prop, continuity]
theorem continuous_of_linearMap (φ : A →ₗ[R] B) : Continuous φ :=
  continuous_of_distribMulActionHom φ.toDistribMulActionHom


variable (R) in
theorem continuous_neg (C : Type*) [AddCommGroup C] [Module R C] [TopologicalSpace C]
    [IsModuleTopology R C] : Continuous (fun a ↦ -a : C → C) :=
  haveI : ContinuousAdd C := IsModuleTopology.toContinuousAdd R C
  continuous_of_linearMap (LinearEquiv.neg R).toLinearMap


variable (R) in
theorem continuousNeg (C : Type*) [AddCommGroup C] [Module R C] [TopologicalSpace C]
    [IsModuleTopology R C] : ContinuousNeg C where
  continuous_neg := continuous_neg R C


variable (R) in
theorem topologicalAddGroup (C : Type*) [AddCommGroup C] [Module R C] [TopologicalSpace C]
    [IsModuleTopology R C] : TopologicalAddGroup C where
      continuous_add := (IsModuleTopology.toContinuousAdd R C).1
      continuous_neg := continuous_neg R C


@[fun_prop, continuity]
theorem continuous_of_ringHom {R A B} [CommSemiring R] [Semiring A] [Algebra R A] [Semiring B]
    [TopologicalSpace R] [TopologicalSpace A] [IsModuleTopology R A] [TopologicalSpace B]
    [TopologicalSemiring B]
    (φ : A →+* B) (hφ : Continuous (φ.comp (algebraMap R A))) : Continuous φ := by
  /-
    R : Type u_4
    A : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSemiring B
    φ : RingHom A B
    hφ : Continuous ⇑(φ.comp (algebraMap R A))
    ⊢ Continuous ⇑φ
  -/
  let inst := Module.compHom B (φ.comp (algebraMap R A))
  /-
    R : Type u_4
    A : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSemiring B
    φ : RingHom A B
    hφ : Continuous ⇑(φ.comp (algebraMap R A))
    inst : Module R B := Module.compHom B (φ.comp (algebraMap R A))
    ⊢ Continuous ⇑φ
  -/
  let φ' : A →ₗ[R] B := ⟨φ, fun r m ↦ by simp [Algebra.smul_def]; rfl⟩
  /-
    R : Type u_4
    A : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSemiring B
    φ : RingHom A B
    hφ : Continuous ⇑(φ.comp (algebraMap R A))
    inst : Module R B := Module.compHom B (φ.comp (algebraMap R A))
    φ' : LinearMap (RingHom.id R) A B := { toAddHom := ↑φ, map_smul' := ⋯ }
    ⊢ Continuous ⇑φ
  -/
  have : ContinuousSMul R B := ⟨(hφ.comp continuous_fst).mul continuous_snd⟩
  /-
    R : Type u_4
    A : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : TopologicalSpace R
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSemiring B
    φ : RingHom A B
    hφ : Continuous ⇑(φ.comp (algebraMap R A))
    inst : Module R B := Module.compHom B (φ.comp (algebraMap R A))
    φ' : LinearMap (RingHom.id R) A B := { toAddHom := ↑φ, map_smul' := ⋯ }
    this : ContinuousSMul R B
    ⊢ Continuous ⇑φ
  -/
  exact continuous_of_linearMap φ'
  /-
    🎉 no goals
  -/


open Topology in
/-- A linear surjection between modules with the module topology is a quotient map.
Equivalently, the pushforward of the module topology along a surjective linear map is
again the module topology. -/
theorem isQuotientMap_of_surjective [τB : TopologicalSpace B] [IsModuleTopology R B]
    {φ : A →ₗ[R] B} (hφ : Function.Surjective φ) :
    IsQuotientMap φ where
  surjective := hφ
  eq_coinduced := by
    -- We need to prove that the topology on B is coinduced from that on A.
    -- First tell the typeclass inference system that A and B are topological groups.
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      ⊢ Eq τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    haveI := topologicalAddGroup R A
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this : TopologicalAddGroup A
      ⊢ Eq τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    haveI := topologicalAddGroup R B
    -- Because φ is linear, it's continuous for the module topologies (by a previous result).
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝ : TopologicalAddGroup A
      this : TopologicalAddGroup B
      ⊢ Eq τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    have this : Continuous φ := continuous_of_linearMap φ
    -- So the coinduced topology is finer than the module topology on B.
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝¹ : TopologicalAddGroup A
      this✝ : TopologicalAddGroup B
      this : Continuous ⇑φ
      ⊢ Eq τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    rw [continuous_iff_coinduced_le] at this
    -- So STP the module topology on B is ≤ the topology coinduced from A
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝¹ : TopologicalAddGroup A
      this✝ : TopologicalAddGroup B
      this : LE.le (TopologicalSpace.coinduced (⇑φ) inst✝⁴) τB
      ⊢ Eq τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    refine le_antisymm ?_ this
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝¹ : TopologicalAddGroup A
      this✝ : TopologicalAddGroup B
      this : LE.le (TopologicalSpace.coinduced (⇑φ) inst✝⁴) τB
      ⊢ LE.le τB (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    rw [eq_moduleTopology R B]
    -- Now let's remove B's topology from the typeclass system
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁷ : Ring R
      A : Type u_2
      inst✝⁶ : AddCommGroup A
      inst✝⁵ : Module R A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : IsModuleTopology R A
      B : Type u_3
      inst✝² : AddCommGroup B
      inst✝¹ : Module R B
      τB : TopologicalSpace B
      inst✝ : IsModuleTopology R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝¹ : TopologicalAddGroup A
      this✝ : TopologicalAddGroup B
      this : LE.le (TopologicalSpace.coinduced (⇑φ) inst✝⁴) τB
      ⊢ LE.le (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inst✝⁴)
    -/
    clear! τB
    -- and replace it with the coinduced topology (which will be the same, but that's what we're
    -- trying to prove). This means we don't have to fight with the typeclass system.
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁶ : Ring R
      A : Type u_2
      inst✝⁵ : AddCommGroup A
      inst✝⁴ : Module R A
      inst✝³ : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommGroup B
      inst✝ : Module R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this : TopologicalAddGroup A
      ⊢ LE.le (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inst✝³)
    -/
    letI : TopologicalSpace B := .coinduced φ inferInstance
    -- With this new topology on `B`, φ is a quotient map by definition,
    -- and hence an open quotient map by a result in the library.
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁶ : Ring R
      A : Type u_2
      inst✝⁵ : AddCommGroup A
      inst✝⁴ : Module R A
      inst✝³ : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommGroup B
      inst✝ : Module R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝ : TopologicalAddGroup A
      this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
      ⊢ LE.le (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inst✝³)
    -/
    have hφo : IsOpenQuotientMap φ := AddMonoidHom.isOpenQuotientMap_of_isQuotientMap ⟨hφ, rfl⟩
    -- We're trying to prove the module topology on B is ≤ the coinduced topology.
    -- But recall that the module topology is the Inf of the topologies on B making addition
    -- and scalar multiplication continuous, so it suffices to prove
    -- that the coinduced topology on B has these properties.
    /-
      R : Type u_1
      τR : TopologicalSpace R
      inst✝⁶ : Ring R
      A : Type u_2
      inst✝⁵ : AddCommGroup A
      inst✝⁴ : Module R A
      inst✝³ : TopologicalSpace A
      inst✝² : IsModuleTopology R A
      B : Type u_3
      inst✝¹ : AddCommGroup B
      inst✝ : Module R B
      φ : LinearMap (RingHom.id R) A B
      hφ : Function.Surjective ⇑φ
      this✝ : TopologicalAddGroup A
      this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
      hφo : IsOpenQuotientMap ⇑φ
      ⊢ LE.le (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inst✝³)
    -/
    refine sInf_le ⟨?_, ?_⟩
    · -- In this branch, we prove that `• : R × B → B` is continuous for the coinduced topology.
      /-
        case refine_1
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        ⊢ ContinuousSMul R B
      -/
      apply ContinuousSMul.mk
      -- We know that `• : R × A → A` is continuous, by assumption.
      /-
        case refine_1.continuous_smul
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
      -/
      obtain ⟨hA⟩ : ContinuousSMul R A := inferInstance
      /- By linearity of φ, this diagram commutes:
        R × A --(•)--> A
          |            |
          |id × φ      |φ
          |            |
         \/            \/
        R × B --(•)--> B
      -/
      have hφ2 : (fun p ↦ p.1 • p.2 : R × B → B) ∘ (Prod.map id φ) =
        φ ∘ (fun p ↦ p.1 • p.2 : R × A → A) := by ext; simp
      -- Furthermore, the identity from R to R is an open quotient map as is `φ`,
      -- so the product `id × φ` is an open quotient map, by a result in the library.
      /-
        case refine_1.continuous_smul.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HSMul.hSMul p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑φ)) (Func …
        ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
      -/
      have hoq : IsOpenQuotientMap (_ : R × A → R × B) := IsOpenQuotientMap.prodMap .id hφo
      -- This is the left map in the diagram. So by a standard fact about open quotient maps,
      -- to prove that the bottom map is continuous, it suffices to prove
      -- that the diagonal map is continuous.
      /-
        case refine_1.continuous_smul.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HSMul.hSMul p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑φ)) (Func …
        hoq : IsOpenQuotientMap (Prod.map id ⇑φ)
        ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
      -/
      rw [← hoq.continuous_comp_iff]
      -- but the diagonal is the composite of the continuous maps `φ` and `• : R × A → A`
      /-
        case refine_1.continuous_smul.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HSMul.hSMul p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑φ)) (Func …
        hoq : IsOpenQuotientMap (Prod.map id ⇑φ)
        ⊢ Continuous (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑φ))
      -/
      rw [hφ2]
      -- so we're done
      /-
        case refine_1.continuous_smul.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HSMul.hSMul p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑φ)) (Func …
        hoq : IsOpenQuotientMap (Prod.map id ⇑φ)
        ⊢ Continuous (Function.comp ⇑φ fun p => HSMul.hSMul p.1 p.2)
      -/
      exact Continuous.comp hφo.continuous hA
      /-
        🎉 no goals
      -/
    · /- In this branch we show that addition is continuous for the coinduced topology on `B`.
        The argument is basically the same, this time using commutativity of
        A × A --(+)--> A
          |            |
          |φ × φ       |φ
          |            |
         \/            \/
        B × B --(+)--> B
      -/
      /-
        case refine_2
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        ⊢ ContinuousAdd B
      -/
      apply ContinuousAdd.mk
      /-
        case refine_2.continuous_add
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        ⊢ Continuous fun p => HAdd.hAdd p.1 p.2
      -/
      obtain ⟨hA⟩ := IsModuleTopology.toContinuousAdd R A
      have hφ2 : (fun p ↦ p.1 + p.2 : B × B → B) ∘ (Prod.map φ φ) =
        φ ∘ (fun p ↦ p.1 + p.2 : A × A → A) := by ext; simp
      /-
        case refine_2.continuous_add.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HAdd.hAdd p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HAdd.hAdd p.1 p.2) (Prod.map ⇑φ ⇑φ)) (Functi …
        ⊢ Continuous fun p => HAdd.hAdd p.1 p.2
      -/
      rw [← (IsOpenQuotientMap.prodMap hφo hφo).continuous_comp_iff, hφ2]
      /-
        case refine_2.continuous_add.mk
        R : Type u_1
        τR : TopologicalSpace R
        inst✝⁶ : Ring R
        A : Type u_2
        inst✝⁵ : AddCommGroup A
        inst✝⁴ : Module R A
        inst✝³ : TopologicalSpace A
        inst✝² : IsModuleTopology R A
        B : Type u_3
        inst✝¹ : AddCommGroup B
        inst✝ : Module R B
        φ : LinearMap (RingHom.id R) A B
        hφ : Function.Surjective ⇑φ
        this✝ : TopologicalAddGroup A
        this : TopologicalSpace B := TopologicalSpace.coinduced (⇑φ) inferInstance
        hφo : IsOpenQuotientMap ⇑φ
        hA : Continuous fun p => HAdd.hAdd p.1 p.2
        hφ2 : Eq (Function.comp (fun p => HAdd.hAdd p.1 p.2) (Prod.map ⇑φ ⇑φ)) (Functi …
        ⊢ Continuous (Function.comp ⇑φ fun p => HAdd.hAdd p.1 p.2)
      -/
      exact Continuous.comp hφo.continuous hA
      /-
        🎉 no goals
      -/


lemma _root_.ModuleTopology.eq_coinduced_of_surjective
    {φ : A →ₗ[R] B} (hφ : Function.Surjective φ) :
    moduleTopology R B = TopologicalSpace.coinduced φ inferInstance := by
  /-
    R : Type u_1
    τR : TopologicalSpace R
    inst✝⁶ : Ring R
    A : Type u_2
    inst✝⁵ : AddCommGroup A
    inst✝⁴ : Module R A
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    B : Type u_3
    inst✝¹ : AddCommGroup B
    inst✝ : Module R B
    φ : LinearMap (RingHom.id R) A B
    hφ : Function.Surjective ⇑φ
    ⊢ Eq (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inferInstance)
  -/
  letI : TopologicalSpace B := moduleTopology R B
  /-
    R : Type u_1
    τR : TopologicalSpace R
    inst✝⁶ : Ring R
    A : Type u_2
    inst✝⁵ : AddCommGroup A
    inst✝⁴ : Module R A
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    B : Type u_3
    inst✝¹ : AddCommGroup B
    inst✝ : Module R B
    φ : LinearMap (RingHom.id R) A B
    hφ : Function.Surjective ⇑φ
    this : TopologicalSpace B := moduleTopology R B
    ⊢ Eq (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inferInstance)
  -/
  haveI : IsModuleTopology R B := ⟨rfl⟩
  /-
    R : Type u_1
    τR : TopologicalSpace R
    inst✝⁶ : Ring R
    A : Type u_2
    inst✝⁵ : AddCommGroup A
    inst✝⁴ : Module R A
    inst✝³ : TopologicalSpace A
    inst✝² : IsModuleTopology R A
    B : Type u_3
    inst✝¹ : AddCommGroup B
    inst✝ : Module R B
    φ : LinearMap (RingHom.id R) A B
    hφ : Function.Surjective ⇑φ
    this✝ : TopologicalSpace B := moduleTopology R B
    this : IsModuleTopology R B
    ⊢ Eq (moduleTopology R B) (TopologicalSpace.coinduced (⇑φ) inferInstance)
  -/
  exact (isQuotientMap_of_surjective hφ).eq_coinduced
  /-
    🎉 no goals
  -/



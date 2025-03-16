theorem lmul_algebraMap (x : R) : Algebra.lmul R A (algebraMap R A x) = Algebra.lsmul R R A x :=
  Eq.symm <| LinearMap.ext <| smul_def x


instance subalgebra (S₀ : Subalgebra R S) : IsScalarTower S₀ S A :=
  of_algebraMap_eq fun _ ↦ rfl


instance subalgebra' (S₀ : Subalgebra R S) : IsScalarTower R S₀ A :=
  @IsScalarTower.of_algebraMap_eq R S₀ A _ _ _ _ _ _ fun _ ↦
    (IsScalarTower.algebraMap_apply R S A _ : _)


/-- Given a tower `A / ↥U / S / R` of algebras, where `U` is an `S`-subalgebra of `A`, reinterpret
`U` as an `R`-subalgebra of `A`. -/
def restrictScalars (U : Subalgebra S A) : Subalgebra R A :=
  { U with
    algebraMap_mem' := fun x ↦ by
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S B
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        U : Subalgebra S A
        x : R
        ⊢ Membership.mem U.carrier ((algebraMap R A) x)
      -/
      rw [algebraMap_apply R S A]
      /-
        R : Type u
        S : Type v
        A : Type w
        B : Type u₁
        M : Type v₁
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra S A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S B
        inst✝² : Algebra R B
        inst✝¹ : IsScalarTower R S A
        inst✝ : IsScalarTower R S B
        U : Subalgebra S A
        x : R
        ⊢ Membership.mem U.carrier ((algebraMap S A) ((algebraMap R S) x))
      -/
      exact U.algebraMap_mem _ }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_restrictScalars {U : Subalgebra S A} : (restrictScalars R U : Set A) = (U : Set A) :=
  rfl


@[simp]
theorem restrictScalars_top : restrictScalars R (⊤ : Subalgebra S A) = ⊤ :=
                              /-
                                R : Type u
                                S : Type v
                                A : Type w
                                inst✝⁶ : CommSemiring R
                                inst✝⁵ : CommSemiring S
                                inst✝⁴ : Semiring A
                                inst✝³ : Algebra R S
                                inst✝² : Algebra S A
                                inst✝¹ : Algebra R A
                                inst✝ : IsScalarTower R S A
                                ⊢ Eq ↑(Subalgebra.restrictScalars R Top.top) ↑Top.top
                              -/
  SetLike.coe_injective <| by dsimp -- Porting note: why does `rfl` not work instead of `by dsimp`?
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem restrictScalars_toSubmodule {U : Subalgebra S A} :
    Subalgebra.toSubmodule (U.restrictScalars R) = U.toSubmodule.restrictScalars R :=
  SetLike.coe_injective rfl


@[simp]
theorem mem_restrictScalars {U : Subalgebra S A} {x : A} : x ∈ restrictScalars R U ↔ x ∈ U :=
  Iff.rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : Subalgebra S A → Subalgebra R A) := fun U V H ↦
                 /-
                   R : Type u
                   S : Type v
                   A : Type w
                   inst✝⁶ : CommSemiring R
                   inst✝⁵ : CommSemiring S
                   inst✝⁴ : Semiring A
                   inst✝³ : Algebra R S
                   inst✝² : Algebra S A
                   inst✝¹ : Algebra R A
                   inst✝ : IsScalarTower R S A
                   U V : Subalgebra S A
                   H : Eq (Subalgebra.restrictScalars R U) (Subalgebra.restrictScalars R V)
                   x : A
                   ⊢ Iff (Membership.mem U x) (Membership.mem V x)
                 -/
  ext fun x ↦ by rw [← mem_restrictScalars R, H, mem_restrictScalars]
                 /-
                   🎉 no goals
                 -/


/-- Produces an `R`-algebra map from `U.restrictScalars R` given an `S`-algebra map from `U`.

This is a special case of `AlgHom.restrictScalars` that can be helpful in elaboration. -/
@[simp]
def ofRestrictScalars (U : Subalgebra S A) (f : U →ₐ[S] B) : U.restrictScalars R →ₐ[R] B :=
  f.restrictScalars R


@[simp]
lemma range_isScalarTower_toAlgHom [CommSemiring R] [CommSemiring A]
    [Algebra R A] (S : Subalgebra R A) :
    LinearMap.range (IsScalarTower.toAlgHom R S A) = Subalgebra.toSubmodule S := by
  /-
    R : Type u
    A : Type w
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    ⊢ Eq (LinearMap.range (IsScalarTower.toAlgHom R (Subtype fun x => Membership.m …
  -/
  ext
  simp only [← Submodule.range_subtype (Subalgebra.toSubmodule S), LinearMap.mem_range,
    IsScalarTower.coe_toAlgHom', Subalgebra.mem_toSubmodule]
  /-
    case h
    R : Type u
    A : Type w
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    S : Subalgebra R A
    x✝ : A
    ⊢ Iff (Exists fun y => Eq ((algebraMap (Subtype fun x => Membership.mem S x) A …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem adjoin_range_toAlgHom (t : Set A) :
    (Algebra.adjoin (toAlgHom R S A).range t).restrictScalars R =
      (Algebra.adjoin S t).restrictScalars R :=
  Subalgebra.ext fun z ↦
    show z ∈ Subsemiring.closure (Set.range (algebraMap (toAlgHom R S A).range A) ∪ t : Set A) ↔
         z ∈ Subsemiring.closure (Set.range (algebraMap S A) ∪ t : Set A) by
      suffices Set.range (algebraMap (toAlgHom R S A).range A) = Set.range (algebraMap S A) by
        rw [this]
      /-
        R : Type u
        S : Type v
        A : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : CommSemiring S
        inst✝⁴ : CommSemiring A
        inst✝³ : Algebra R S
        inst✝² : Algebra S A
        inst✝¹ : Algebra R A
        inst✝ : IsScalarTower R S A
        t : Set A
        z : A
        ⊢ Eq (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem (IsScalarTower.t …
      -/
      ext z
      /-
        case h
        R : Type u
        S : Type v
        A : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : CommSemiring S
        inst✝⁴ : CommSemiring A
        inst✝³ : Algebra R S
        inst✝² : Algebra S A
        inst✝¹ : Algebra R A
        inst✝ : IsScalarTower R S A
        t : Set A
        z✝ z : A
        ⊢ Iff (Membership.mem (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem …
      -/
      exact ⟨fun ⟨⟨_, y, h1⟩, h2⟩ ↦ ⟨y, h2 ▸ h1⟩, fun ⟨y, hy⟩ ↦ ⟨⟨z, y, hy⟩, rfl⟩⟩
      /-
        🎉 no goals
      -/



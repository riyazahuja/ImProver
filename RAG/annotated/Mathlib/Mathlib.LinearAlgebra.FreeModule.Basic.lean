/-- `Module.Free R M` is the statement that the `R`-module `M` is free. -/
class Module.Free : Prop where
  exists_basis : Nonempty <| (I : Type v) × Basis I R M


theorem Module.free_iff_set : Module.Free R M ↔ ∃ S : Set M, Nonempty (Basis S R M) :=
  ⟨fun h => ⟨Set.range h.exists_basis.some.2, ⟨Basis.reindexRange h.exists_basis.some.2⟩⟩,
    fun ⟨S, hS⟩ => ⟨nonempty_sigma.2 ⟨S, hS⟩⟩⟩


/-- If `M` fits in universe `w`, then freeness is equivalent to existence of a basis in that
universe.

Note that if `M` does not fit in `w`, the reverse direction of this implication is still true as
`Module.Free.of_basis`. -/
theorem Module.free_def [Small.{w,v} M] :
    Module.Free R M ↔ ∃ I : Type w, Nonempty (Basis I R M) :=
  ⟨fun h =>
    ⟨Shrink (Set.range h.exists_basis.some.2),
      ⟨(Basis.reindexRange h.exists_basis.some.2).reindex (equivShrink _)⟩⟩,
    fun h => ⟨(nonempty_sigma.2 h).map fun ⟨_, b⟩ => ⟨Set.range b, b.reindexRange⟩⟩⟩


theorem Module.Free.of_basis {ι : Type w} (b : Basis ι R M) : Module.Free R M :=
  (Module.free_def R M).2 ⟨Set.range b, ⟨b.reindexRange⟩⟩


/-- If `Module.Free R M` then `ChooseBasisIndex R M` is the `ι` which indexes the basis
  `ι → M`. Note that this is defined such that this type is finite if `R` is trivial. -/
def ChooseBasisIndex : Type _ :=
  ((Module.free_iff_set R M).mp ‹_›).choose


/-- There is no hope of computing this, but we add the instance anyway to avoid fumbling with
`open scoped Classical`. -/
noncomputable instance : DecidableEq (ChooseBasisIndex R M) := Classical.decEq _


/-- If `Module.Free R M` then `chooseBasis : ι → M` is the basis.
Here `ι = ChooseBasisIndex R M`. -/
noncomputable def chooseBasis : Basis (ChooseBasisIndex R M) R M :=
  ((Module.free_iff_set R M).mp ‹_›).choose_spec.some


/-- The isomorphism `M ≃ₗ[R] (ChooseBasisIndex R M →₀ R)`. -/
noncomputable def repr : M ≃ₗ[R] ChooseBasisIndex R M →₀ R :=
  (chooseBasis R M).repr


/-- The universal property of free modules: giving a function `(ChooseBasisIndex R M) → N`, for `N`
an `R`-module, is the same as giving an `R`-linear map `M →ₗ[R] N`.

This definition is parameterized over an extra `Semiring S`,
such that `SMulCommClass R S M'` holds.
If `R` is commutative, you can set `S := R`; if `R` is not commutative,
you can recover an `AddEquiv` by setting `S := ℕ`.
See library note [bundled maps over different rings]. -/
noncomputable def constr {S : Type z} [Semiring S] [Module S N] [SMulCommClass R S N] :
    (ChooseBasisIndex R M → N) ≃ₗ[S] M →ₗ[R] N :=
  Basis.constr (chooseBasis R M) S


instance (priority := 100) noZeroSMulDivisors [NoZeroDivisors R] : NoZeroSMulDivisors R M :=
  let ⟨⟨_, b⟩⟩ := exists_basis (R := R) (M := M)
  b.noZeroSMulDivisors


instance [Nontrivial M] : Nonempty (Module.Free.ChooseBasisIndex R M) :=
  (Module.Free.chooseBasis R M).index_nonempty


theorem infinite [Infinite R] [Nontrivial M] : Infinite M :=
  (Equiv.infinite_iff (chooseBasis R M).repr.toEquiv).mpr Finsupp.infinite_of_right


theorem of_equiv (e : M ≃ₗ[R] N) : Module.Free R N :=
  of_basis <| (chooseBasis R M).map e


/-- A variation of `of_equiv`: the assumption `Module.Free R P` here is explicit rather than an
instance. -/
theorem of_equiv' {P : Type v} [AddCommMonoid P] [Module R P] (_ : Module.Free R P)
    (e : P ≃ₗ[R] N) : Module.Free R N :=
  of_equiv e


attribute [local instance] RingHomInvPair.of_ringEquiv in
lemma of_ringEquiv {R R' M M'} [Semiring R] [AddCommMonoid M] [Module R M]
    [Semiring R'] [AddCommMonoid M'] [Module R' M']
    (e₁ : R ≃+* R') (e₂ : M ≃ₛₗ[RingHomClass.toRingHom e₁] M') [Module.Free R M] :
    Module.Free R' M' := by
  /-
    R : Type u_2
    R' : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring R'
    inst✝² : AddCommMonoid M'
    inst✝¹ : Module R' M'
    e₁ : RingEquiv R R'
    e₂ : LinearEquiv (↑e₁) M M'
    inst✝ : Module.Free R M
    ⊢ Module.Free R' M'
  -/
  let I := Module.Free.ChooseBasisIndex R M
  /-
    R : Type u_2
    R' : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring R'
    inst✝² : AddCommMonoid M'
    inst✝¹ : Module R' M'
    e₁ : RingEquiv R R'
    e₂ : LinearEquiv (↑e₁) M M'
    inst✝ : Module.Free R M
    I : Type u_4 := Module.Free.ChooseBasisIndex R M
    ⊢ Module.Free R' M'
  -/
  obtain ⟨e₃ : M ≃ₗ[R] I →₀ R⟩ := Module.Free.chooseBasis R M
  let e : M' ≃+ (I →₀ R') :=
    (e₂.symm.trans e₃).toAddEquiv.trans (Finsupp.mapRange.addEquiv (α := I) e₁.toAddEquiv)
  /-
    case ofRepr
    R : Type u_2
    R' : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring R'
    inst✝² : AddCommMonoid M'
    inst✝¹ : Module R' M'
    e₁ : RingEquiv R R'
    e₂ : LinearEquiv (↑e₁) M M'
    inst✝ : Module.Free R M
    I : Type u_4 := Module.Free.ChooseBasisIndex R M
    e₃ : LinearEquiv (RingHom.id R) M (Finsupp I R)
    e : AddEquiv M' (Finsupp I R') := (e₂.symm.trans e₃).toAddEquiv.trans (Finsupp …
    ⊢ Module.Free R' M'
  -/
  have he (x) : e x = Finsupp.mapRange.addEquiv (α := I) e₁.toAddEquiv (e₃ (e₂.symm x)) := rfl
  let e' : M' ≃ₗ[R'] (I →₀ R') :=
    { __ := e, map_smul' := fun m x ↦ Finsupp.ext fun i ↦ by simp [he, map_smulₛₗ] }
  /-
    case ofRepr
    R : Type u_2
    R' : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring R'
    inst✝² : AddCommMonoid M'
    inst✝¹ : Module R' M'
    e₁ : RingEquiv R R'
    e₂ : LinearEquiv (↑e₁) M M'
    inst✝ : Module.Free R M
    I : Type u_4 := Module.Free.ChooseBasisIndex R M
    e₃ : LinearEquiv (RingHom.id R) M (Finsupp I R)
    e : AddEquiv M' (Finsupp I R') := (e₂.symm.trans e₃).toAddEquiv.trans (Finsupp …
    he : ∀ (x : M'), Eq (e x) ((Finsupp.mapRange.addEquiv e₁.toAddEquiv) (e₃ (e₂.s …
    e' : LinearEquiv (RingHom.id R') M' (Finsupp I R') :=
      let __spread.0 := e;
      { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯, invFun := __spre …
    ⊢ Module.Free R' M'
  -/
  exact of_basis (.ofRepr e')
  /-
    🎉 no goals
  -/


attribute [local instance] RingHomInvPair.of_ringEquiv in
lemma iff_of_ringEquiv {R R' M M'} [Semiring R] [AddCommMonoid M] [Module R M]
    [Semiring R'] [AddCommMonoid M'] [Module R' M']
    (e₁ : R ≃+* R') (e₂ : M ≃ₛₗ[RingHomClass.toRingHom e₁] M') :
    Module.Free R M ↔ Module.Free R' M' :=
  ⟨fun _ ↦ of_ringEquiv e₁ e₂, fun _ ↦ of_ringEquiv e₁.symm e₂.symm⟩


/-- The module structure provided by `Semiring.toModule` is free. -/
instance self : Module.Free R R :=
  of_basis (Basis.singleton Unit R)


instance prod [Module.Free R N] : Module.Free R (M × N) :=
  of_basis <| (chooseBasis R M).prod (chooseBasis R N)


/-- The product of finitely many free modules is free. -/
instance pi (M : ι → Type*) [Finite ι] [∀ i : ι, AddCommMonoid (M i)] [∀ i : ι, Module R (M i)]
    [∀ i : ι, Module.Free R (M i)] : Module.Free R (∀ i, M i) :=
  let ⟨_⟩ := nonempty_fintype ι
  of_basis <| Pi.basis fun i => chooseBasis R (M i)


/-- The module of finite matrices is free. -/
instance matrix {m n : Type*} [Finite m] [Finite n] : Module.Free R (Matrix m n M) :=
  Module.Free.pi R _


instance ulift [Free R M] : Free R (ULift M) := of_equiv ULift.moduleEquiv.symm


/-- The product of finitely many free modules is free (non-dependent version to help with typeclass
search). -/
instance function [Finite ι] : Module.Free R (ι → M) :=
  Free.pi _ _


instance finsupp : Module.Free R (ι →₀ M) :=
  of_basis (Finsupp.basis fun _ => chooseBasis R M)


instance (priority := 100) of_subsingleton [Subsingleton N] : Module.Free R N :=
  of_basis.{u,z,z} (Basis.empty N : Basis PEmpty R N)


instance (priority := 100) of_subsingleton' [Subsingleton R] : Module.Free R N :=
  letI := Module.subsingleton R N
  Module.Free.of_subsingleton R N


instance dfinsupp {ι : Type*} (M : ι → Type*) [∀ i : ι, AddCommMonoid (M i)]
    [∀ i : ι, Module R (M i)] [∀ i : ι, Module.Free R (M i)] : Module.Free R (Π₀ i, M i) :=
  of_basis <| DFinsupp.basis fun i => chooseBasis R (M i)


instance directSum {ι : Type*} (M : ι → Type*) [∀ i : ι, AddCommMonoid (M i)]
    [∀ i : ι, Module R (M i)] [∀ i : ι, Module.Free R (M i)] : Module.Free R (⨁ i, M i) :=
  Module.Free.dfinsupp R M


instance tensor : Module.Free S (M ⊗[R] N) :=
  let ⟨bM⟩ := exists_basis (R := S) (M := M)
  let ⟨bN⟩ := exists_basis (R := R) (M := N)
  of_basis (bM.2.tensorProduct bN.2)



private theorem free_and_finite_fin (n : ℕ) (N : Fin n → Type*) [∀ i, AddCommGroup (N i)]
    [∀ i, Module R (N i)] [∀ i, Module.Finite R (N i)] [∀ i, Module.Free R (N i)] :
    Module.Free R (MultilinearMap R N M₂) ∧ Module.Finite R (MultilinearMap R N M₂) := by
  /-
    R : Type u_2
    M₂ : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module.Finite R M₂
    inst✝⁴ : Module.Free R M₂
    n : Nat
    N : Fin n → Type u_5
    inst✝³ : (i : Fin n) → AddCommGroup (N i)
    inst✝² : (i : Fin n) → Module R (N i)
    inst✝¹ : ∀ (i : Fin n), Module.Finite R (N i)
    inst✝ : ∀ (i : Fin n), Module.Free R (N i)
    ⊢ And (Module.Free R (MultilinearMap R N M₂)) (Module.Finite R (MultilinearMap …
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u_2
      M₂ : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module.Finite R M₂
      inst✝⁴ : Module.Free R M₂
      N : Fin 0 → Type u_5
      inst✝³ : (i : Fin 0) → AddCommGroup (N i)
      inst✝² : (i : Fin 0) → Module R (N i)
      inst✝¹ : ∀ (i : Fin 0), Module.Finite R (N i)
      inst✝ : ∀ (i : Fin 0), Module.Free R (N i)
      ⊢ And (Module.Free R (MultilinearMap R N M₂)) (Module.Finite R (MultilinearMap …
    -/
  · haveI : IsEmpty (Fin Nat.zero) := inferInstanceAs (IsEmpty (Fin 0))
    exact
      ⟨Module.Free.of_equiv (constLinearEquivOfIsEmpty R R N M₂),
        Module.Finite.equiv (constLinearEquivOfIsEmpty R R N M₂)⟩
  · suffices
      Module.Free R (N 0 →ₗ[R] MultilinearMap R (fun i : Fin n => N i.succ) M₂) ∧
        Module.Finite R (N 0 →ₗ[R] MultilinearMap R (fun i : Fin n => N i.succ) M₂) by
      cases this
      exact
        ⟨Module.Free.of_equiv (multilinearCurryLeftEquiv R N M₂).symm,
          Module.Finite.equiv (multilinearCurryLeftEquiv R N M₂).symm⟩
    /-
      case succ
      R : Type u_2
      M₂ : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module.Finite R M₂
      inst✝⁴ : Module.Free R M₂
      n : Nat
      ih : ∀ (N : Fin n → Type u_5) [inst : (i : Fin n) → AddCommGroup (N i)] [inst_ …
      N : Fin (HAdd.hAdd n 1) → Type u_5
      inst✝³ : (i : Fin (HAdd.hAdd n 1)) → AddCommGroup (N i)
      inst✝² : (i : Fin (HAdd.hAdd n 1)) → Module R (N i)
      inst✝¹ : ∀ (i : Fin (HAdd.hAdd n 1)), Module.Finite R (N i)
      inst✝ : ∀ (i : Fin (HAdd.hAdd n 1)), Module.Free R (N i)
      ⊢ And (Module.Free R (LinearMap (RingHom.id R) (N 0) (MultilinearMap R (fun i  …
    -/
    cases ih fun i => N i.succ
    /-
      case succ.intro
      R : Type u_2
      M₂ : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module.Finite R M₂
      inst✝⁴ : Module.Free R M₂
      n : Nat
      ih : ∀ (N : Fin n → Type u_5) [inst : (i : Fin n) → AddCommGroup (N i)] [inst_ …
      N : Fin (HAdd.hAdd n 1) → Type u_5
      inst✝³ : (i : Fin (HAdd.hAdd n 1)) → AddCommGroup (N i)
      inst✝² : (i : Fin (HAdd.hAdd n 1)) → Module R (N i)
      inst✝¹ : ∀ (i : Fin (HAdd.hAdd n 1)), Module.Finite R (N i)
      inst✝ : ∀ (i : Fin (HAdd.hAdd n 1)), Module.Free R (N i)
      left✝ : Module.Free R (MultilinearMap R (fun i => N i.succ) M₂)
      right✝ : Module.Finite R (MultilinearMap R (fun i => N i.succ) M₂)
      ⊢ And (Module.Free R (LinearMap (RingHom.id R) (N 0) (MultilinearMap R (fun i  …
    -/
    exact ⟨Module.Free.linearMap _ _ _ _, Module.Finite.linearMap _ _ _ _⟩
    /-
      🎉 no goals
    -/


private theorem free_and_finite :
    Module.Free R (MultilinearMap R M₁ M₂) ∧ Module.Finite R (MultilinearMap R M₁ M₂) := by
  /-
    ι : Type u_1
    R : Type u_2
    M₂ : Type u_3
    M₁ : ι → Type u_4
    inst✝⁹ : Finite ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module.Finite R M₂
    inst✝⁴ : Module.Free R M₂
    inst✝³ : (i : ι) → AddCommGroup (M₁ i)
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : ∀ (i : ι), Module.Finite R (M₁ i)
    inst✝ : ∀ (i : ι), Module.Free R (M₁ i)
    ⊢ And (Module.Free R (MultilinearMap R M₁ M₂)) (Module.Finite R (MultilinearMa …
  -/
  cases nonempty_fintype ι
  have := @free_and_finite_fin R M₂ _ _ _ _ _ (Fintype.card ι)
    (fun x => M₁ ((Fintype.equivFin ι).symm x))
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M₂ : Type u_3
    M₁ : ι → Type u_4
    inst✝⁹ : Finite ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module.Finite R M₂
    inst✝⁴ : Module.Free R M₂
    inst✝³ : (i : ι) → AddCommGroup (M₁ i)
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : ∀ (i : ι), Module.Finite R (M₁ i)
    inst✝ : ∀ (i : ι), Module.Free R (M₁ i)
    val✝ : Fintype ι
    this : ∀ [inst : (i : Fin (Fintype.card ι)) → AddCommGroup (M₁ ((Fintype.equiv …
    ⊢ And (Module.Free R (MultilinearMap R M₁ M₂)) (Module.Finite R (MultilinearMa …
  -/
  cases' this with l r
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M₂ : Type u_3
    M₁ : ι → Type u_4
    inst✝⁹ : Finite ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module.Finite R M₂
    inst✝⁴ : Module.Free R M₂
    inst✝³ : (i : ι) → AddCommGroup (M₁ i)
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : ∀ (i : ι), Module.Finite R (M₁ i)
    inst✝ : ∀ (i : ι), Module.Free R (M₁ i)
    val✝ : Fintype ι
    this : ∀ [inst : (i : Fin (Fintype.card ι)) → AddCommGroup (M₁ ((Fintype.equiv …
    l : Module.Free R (MultilinearMap R (fun x => M₁ ((Fintype.equivFin ι).symm x) …
    r : Module.Finite R (MultilinearMap R (fun x => M₁ ((Fintype.equivFin ι).symm  …
    ⊢ And (Module.Free R (MultilinearMap R M₁ M₂)) (Module.Finite R (MultilinearMa …
  -/
  have e := domDomCongrLinearEquiv' R R M₁ M₂ (Fintype.equivFin ι)
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_2
    M₂ : Type u_3
    M₁ : ι → Type u_4
    inst✝⁹ : Finite ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : Module.Finite R M₂
    inst✝⁴ : Module.Free R M₂
    inst✝³ : (i : ι) → AddCommGroup (M₁ i)
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : ∀ (i : ι), Module.Finite R (M₁ i)
    inst✝ : ∀ (i : ι), Module.Free R (M₁ i)
    val✝ : Fintype ι
    this : ∀ [inst : (i : Fin (Fintype.card ι)) → AddCommGroup (M₁ ((Fintype.equiv …
    l : Module.Free R (MultilinearMap R (fun x => M₁ ((Fintype.equivFin ι).symm x) …
    r : Module.Finite R (MultilinearMap R (fun x => M₁ ((Fintype.equivFin ι).symm  …
    e : LinearEquiv (RingHom.id R) (MultilinearMap R M₁ M₂) (MultilinearMap R (fun …
    ⊢ And (Module.Free R (MultilinearMap R M₁ M₂)) (Module.Finite R (MultilinearMa …
  -/
  exact ⟨Module.Free.of_equiv e.symm, Module.Finite.equiv e.symm⟩
  /-
    🎉 no goals
  -/


instance _root_.Module.Finite.multilinearMap : Module.Finite R (MultilinearMap R M₁ M₂) :=
  free_and_finite.2


instance _root_.Module.Free.multilinearMap : Module.Free R (MultilinearMap R M₁ M₂) :=
  free_and_finite.1



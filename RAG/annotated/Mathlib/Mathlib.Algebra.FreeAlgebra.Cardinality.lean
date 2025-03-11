@[simp]
theorem cardinalMk_eq_max_lift [Nonempty X] [Nontrivial R] :
    #(FreeAlgebra R X) = Cardinal.lift.{v} #R ⊔ Cardinal.lift.{u} #X ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    X : Type v
    inst✝¹ : Nonempty X
    inst✝ : Nontrivial R
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u} ( …
  -/
  have hX := mk_freeMonoid X
  /-
    R : Type u
    inst✝² : CommSemiring R
    X : Type v
    inst✝¹ : Nonempty X
    inst✝ : Nontrivial R
    hX : Eq (Cardinal.mk (FreeMonoid X)) (Max.max (Cardinal.mk X) Cardinal.aleph0)
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u} ( …
  -/
  haveI : Infinite (FreeMonoid X) := infinite_iff.2 (by simp [hX])
  rw [equivMonoidAlgebraFreeMonoid.toEquiv.cardinal_eq, MonoidAlgebra,
    mk_finsupp_lift_of_infinite, hX, lift_max, lift_aleph0, sup_comm, ← sup_assoc]


@[simp]
theorem cardinalMk_eq_lift [IsEmpty X] : #(FreeAlgebra R X) = Cardinal.lift.{v} #R := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    X : Type v
    inst✝ : IsEmpty X
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Cardinal.lift.{v, u} (Cardinal.mk R))
  -/
  have := lift_mk_eq'.2 ⟨show (FreeMonoid X →₀ R) ≃ R from Equiv.finsuppUnique⟩
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    X : Type v
    inst✝ : IsEmpty X
    this : Eq (Cardinal.lift.{u, max u v} (Cardinal.mk (Finsupp (FreeMonoid X) R)) …
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Cardinal.lift.{v, u} (Cardinal.mk R))
  -/
  rw [lift_id'.{u, v}, lift_umax] at this
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    X : Type v
    inst✝ : IsEmpty X
    this : Eq (Cardinal.mk (Finsupp (FreeMonoid X) R)) (Cardinal.lift.{v, u} (Card …
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Cardinal.lift.{v, u} (Cardinal.mk R))
  -/
  rwa [equivMonoidAlgebraFreeMonoid.toEquiv.cardinal_eq, MonoidAlgebra]
  /-
    🎉 no goals
  -/


@[nontriviality]
theorem cardinalMk_eq_one [Subsingleton R] : #(FreeAlgebra R X) = 1 := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    X : Type v
    inst✝ : Subsingleton R
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) 1
  -/
  rw [equivMonoidAlgebraFreeMonoid.toEquiv.cardinal_eq, MonoidAlgebra, mk_eq_one]
  /-
    🎉 no goals
  -/


theorem cardinalMk_le_max_lift :
    #(FreeAlgebra R X) ≤ Cardinal.lift.{v} #R ⊔ Cardinal.lift.{u} #X ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    X : Type v
    ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      inst✝ : CommSemiring R
      X : Type v
      h✝ : Subsingleton R
      ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u …
    -/
  · exact (cardinalMk_eq_one R X).trans_le (le_max_of_le_right one_le_aleph0)
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : CommSemiring R
    X : Type v
    h✝ : Nontrivial R
    ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u …
  -/
  cases isEmpty_or_nonempty X
    /-
      case inr.inl
      R : Type u
      inst✝ : CommSemiring R
      X : Type v
      h✝¹ : Nontrivial R
      h✝ : IsEmpty X
      ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u …
    -/
  · exact (cardinalMk_eq_lift R X).trans_le (le_max_of_le_left <| le_max_left _ _)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u
      inst✝ : CommSemiring R
      X : Type v
      h✝¹ : Nontrivial R
      h✝ : Nonempty X
      ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.lift.{v, u …
    -/
  · exact (cardinalMk_eq_max_lift R X).le
    /-
      🎉 no goals
    -/


theorem cardinalMk_eq_max [Nonempty X] [Nontrivial R] : #(FreeAlgebra R X) = #R ⊔ #X ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    X : Type u
    inst✝¹ : Nonempty X
    inst✝ : Nontrivial R
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.mk R) (Cardin …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem cardinalMk_eq [IsEmpty X] : #(FreeAlgebra R X) = #R := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    X : Type u
    inst✝ : IsEmpty X
    ⊢ Eq (Cardinal.mk (FreeAlgebra R X)) (Cardinal.mk R)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem cardinalMk_le_max : #(FreeAlgebra R X) ≤ #R ⊔ #X ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    X : Type u
    ⊢ LE.le (Cardinal.mk (FreeAlgebra R X)) (Max.max (Max.max (Cardinal.mk R) (Car …
  -/
  simpa using cardinalMk_le_max_lift R X
  /-
    🎉 no goals
  -/


theorem lift_cardinalMk_adjoin_le {A : Type v} [Semiring A] [Algebra R A] (s : Set A) :
    lift.{u} #(adjoin R s) ≤ lift.{v} #R ⊔ lift.{u} #s ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (A …
  -/
  have H := mk_range_le_lift (f := FreeAlgebra.lift R ((↑) : s → A))
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    H : LE.le (Cardinal.lift.{max u v, v} (Cardinal.mk ↑(Set.range ⇑((FreeAlgebra. …
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (A …
  -/
  rw [lift_umax, lift_id'.{v, u}] at H
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    H : LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.range ⇑((FreeAlgebra.lift R …
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (A …
  -/
  rw [Algebra.adjoin_eq_range_freeAlgebra_lift]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    H : LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.range ⇑((FreeAlgebra.lift R …
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => Membership.mem (( …
  -/
  exact H.trans (FreeAlgebra.cardinalMk_le_max_lift R s)
  /-
    🎉 no goals
  -/


theorem cardinalMk_adjoin_le {A : Type u} [Semiring A] [Algebra R A] (s : Set A) :
    #(adjoin R s) ≤ #R ⊔ #s ⊔ ℵ₀ := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type u
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ LE.le (Cardinal.mk (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)) …
  -/
  simpa using lift_cardinalMk_adjoin_le R s
  /-
    🎉 no goals
  -/



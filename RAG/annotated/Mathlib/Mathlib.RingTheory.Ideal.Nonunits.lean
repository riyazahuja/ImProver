/-- The set of non-invertible elements of a monoid. -/
def nonunits (α : Type*) [Monoid α] : Set α :=
  { a | ¬IsUnit a }


@[simp]
theorem mem_nonunits_iff [Monoid α] : a ∈ nonunits α ↔ ¬IsUnit a :=
  Iff.rfl


theorem mul_mem_nonunits_right [CommMonoid α] : b ∈ nonunits α → a * b ∈ nonunits α :=
  mt isUnit_of_mul_isUnit_right


theorem mul_mem_nonunits_left [CommMonoid α] : a ∈ nonunits α → a * b ∈ nonunits α :=
  mt isUnit_of_mul_isUnit_left


theorem zero_mem_nonunits [Semiring α] : 0 ∈ nonunits α ↔ (0 : α) ≠ 1 :=
  not_congr isUnit_zero_iff


@[simp 1001] -- increased priority to appease `simpNF`
theorem one_not_mem_nonunits [Monoid α] : (1 : α) ∉ nonunits α :=
  not_not_intro isUnit_one

-- Porting note : as this can be proved by other `simp` lemmas, this is marked as high priority.

@[simp (high)]
theorem map_mem_nonunits_iff [Monoid α] [Monoid β] [FunLike F α β] [MonoidHomClass F α β] (f : F)
    [IsLocalHom f] (a) : f a ∈ nonunits β ↔ a ∈ nonunits α :=
  ⟨fun h ha => h <| ha.map f, fun h ha => h <| ha.of_map⟩


theorem coe_subset_nonunits [Semiring α] {I : Ideal α} (h : I ≠ ⊤) : (I : Set α) ⊆ nonunits α :=
  fun _x hx hu => h <| I.eq_top_of_isUnit_mem hx hu


theorem exists_max_ideal_of_mem_nonunits [CommSemiring α] (h : a ∈ nonunits α) :
    ∃ I : Ideal α, I.IsMaximal ∧ a ∈ I := by
  have : Ideal.span ({a} : Set α) ≠ ⊤ := by
    intro H
    rw [Ideal.span_singleton_eq_top] at H
    contradiction
  /-
    α : Type u_2
    a : α
    inst✝ : CommSemiring α
    h : Membership.mem (nonunits α) a
    this : Ne (Ideal.span (Singleton.singleton a)) Top.top
    ⊢ Exists fun I => And I.IsMaximal (Membership.mem I a)
  -/
  rcases Ideal.exists_le_maximal _ this with ⟨I, Imax, H⟩
  /-
    case intro.intro
    α : Type u_2
    a : α
    inst✝ : CommSemiring α
    h : Membership.mem (nonunits α) a
    this : Ne (Ideal.span (Singleton.singleton a)) Top.top
    I : Ideal α
    Imax : I.IsMaximal
    H : LE.le (Ideal.span (Singleton.singleton a)) I
    ⊢ Exists fun I => And I.IsMaximal (Membership.mem I a)
  -/
  use I, Imax
  /-
    case right
    α : Type u_2
    a : α
    inst✝ : CommSemiring α
    h : Membership.mem (nonunits α) a
    this : Ne (Ideal.span (Singleton.singleton a)) Top.top
    I : Ideal α
    Imax : I.IsMaximal
    H : LE.le (Ideal.span (Singleton.singleton a)) I
    ⊢ Membership.mem I a
  -/
  apply H
  /-
    case right.a
    α : Type u_2
    a : α
    inst✝ : CommSemiring α
    h : Membership.mem (nonunits α) a
    this : Ne (Ideal.span (Singleton.singleton a)) Top.top
    I : Ideal α
    Imax : I.IsMaximal
    H : LE.le (Ideal.span (Singleton.singleton a)) I
    ⊢ Membership.mem (Ideal.span (Singleton.singleton a)) a
  -/
  apply Ideal.subset_span
  /-
    case right.a.a
    α : Type u_2
    a : α
    inst✝ : CommSemiring α
    h : Membership.mem (nonunits α) a
    this : Ne (Ideal.span (Singleton.singleton a)) Top.top
    I : Ideal α
    Imax : I.IsMaximal
    H : LE.le (Ideal.span (Singleton.singleton a)) I
    ⊢ Membership.mem (Singleton.singleton a) a
  -/
  exact Set.mem_singleton a
  /-
    🎉 no goals
  -/


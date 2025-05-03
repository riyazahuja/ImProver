attribute [local simp] pow_le_pow_iff_right₀ one_lt_two inv_le_inv₀ zero_le_two zero_lt_two


open Classical in
/-- In a product space `Π n, E n`, then `firstDiff x y` is the first index at which `x` and `y`
differ. If `x = y`, then by convention we set `firstDiff x x = 0`. -/
irreducible_def firstDiff (x y : ∀ n, E n) : ℕ :=
  if h : x ≠ y then Nat.find (ne_iff.1 h) else 0


theorem apply_firstDiff_ne {x y : ∀ n, E n} (h : x ≠ y) :
    x (firstDiff x y) ≠ y (firstDiff x y) := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    h : Ne x y
    ⊢ Ne (x (PiNat.firstDiff x y)) (y (PiNat.firstDiff x y))
  -/
  rw [firstDiff_def, dif_pos h]
  classical
  exact Nat.find_spec (ne_iff.1 h)


theorem apply_eq_of_lt_firstDiff {x y : ∀ n, E n} {n : ℕ} (hn : n < firstDiff x y) : x n = y n := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hn : LT.lt n (PiNat.firstDiff x y)
    ⊢ Eq (x n) (y n)
  -/
  rw [firstDiff_def] at hn
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hn : LT.lt n (dite (Ne x y) (fun h => Nat.find ⋯) fun h => 0)
    ⊢ Eq (x n) (y n)
  -/
  split_ifs at hn with h
    /-
      case pos
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      h : Ne x y
      hn : LT.lt n (Nat.find ⋯)
      ⊢ Eq (x n) (y n)
    -/
  · convert Nat.find_min (ne_iff.1 h) hn
    /-
      case a
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      h : Ne x y
      hn : LT.lt n (Nat.find ⋯)
      ⊢ Iff (Eq (x n) (y n)) (Not (Ne (x n) (y n)))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      h : Not (Ne x y)
      hn : LT.lt n 0
      ⊢ Eq (x n) (y n)
    -/
  · exact (not_lt_zero' hn).elim
    /-
      🎉 no goals
    -/


theorem firstDiff_comm (x y : ∀ n, E n) : firstDiff x y = firstDiff y x := by
  classical
  simp only [firstDiff_def, ne_comm]


theorem min_firstDiff_le (x y z : ∀ n, E n) (h : x ≠ z) :
    min (firstDiff x y) (firstDiff y z) ≤ firstDiff x z := by
  /-
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    h : Ne x z
    ⊢ LE.le (Min.min (PiNat.firstDiff x y) (PiNat.firstDiff y z)) (PiNat.firstDiff …
  -/
  by_contra! H
  /-
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    h : Ne x z
    H : LT.lt (PiNat.firstDiff x z) (Min.min (PiNat.firstDiff x y) (PiNat.firstDif …
    ⊢ False
  -/
  rw [lt_min_iff] at H
  /-
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    h : Ne x z
    H : And (LT.lt (PiNat.firstDiff x z) (PiNat.firstDiff x y)) (LT.lt (PiNat.firs …
    ⊢ False
  -/
  refine apply_firstDiff_ne h ?_
  calc
    x (firstDiff x z) = y (firstDiff x z) := apply_eq_of_lt_firstDiff H.1
    _ = z (firstDiff x z) := apply_eq_of_lt_firstDiff H.2


/-- In a product space `Π n, E n`, the cylinder set of length `n` around `x`, denoted
`cylinder x n`, is the set of sequences `y` that coincide with `x` on the first `n` symbols, i.e.,
such that `y i = x i` for all `i < n`.
-/
def cylinder (x : ∀ n, E n) (n : ℕ) : Set (∀ n, E n) :=
  { y | ∀ i, i < n → y i = x i }


theorem cylinder_eq_pi (x : ∀ n, E n) (n : ℕ) :
    cylinder x n = Set.pi (Finset.range n : Set ℕ) fun i : ℕ => {x i} := by
  /-
    E : Nat → Type u_1
    x : (n : Nat) → E n
    n : Nat
    ⊢ Eq (PiNat.cylinder x n) ((↑(Finset.range n)).pi fun i => Singleton.singleton …
  -/
  ext y
  /-
    case h
    E : Nat → Type u_1
    x : (n : Nat) → E n
    n : Nat
    y : (n : Nat) → E n
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (Membership.mem ((↑(Finset.range …
  -/
  simp [cylinder]
  /-
    🎉 no goals
  -/


@[simp]
                                                                 /-
                                                                   E : Nat → Type u_1
                                                                   x : (n : Nat) → E n
                                                                   ⊢ Eq (PiNat.cylinder x 0) Set.univ
                                                                 -/
theorem cylinder_zero (x : ∀ n, E n) : cylinder x 0 = univ := by simp [cylinder_eq_pi]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem cylinder_anti (x : ∀ n, E n) {m n : ℕ} (h : m ≤ n) : cylinder x n ⊆ cylinder x m :=
  fun _y hy i hi => hy i (hi.trans_le h)


@[simp]
theorem mem_cylinder_iff {x y : ∀ n, E n} {n : ℕ} : y ∈ cylinder x n ↔ ∀ i < n, y i = x i :=
  Iff.rfl


                                                                          /-
                                                                            E : Nat → Type u_1
                                                                            x : (n : Nat) → E n
                                                                            n : Nat
                                                                            ⊢ Membership.mem (PiNat.cylinder x n) x
                                                                          -/
theorem self_mem_cylinder (x : ∀ n, E n) (n : ℕ) : x ∈ cylinder x n := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem mem_cylinder_iff_eq {x y : ∀ n, E n} {n : ℕ} :
    y ∈ cylinder x n ↔ cylinder y n = cylinder x n := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (Eq (PiNat.cylinder y n) (PiNat. …
  -/
  constructor
    /-
      case mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      ⊢ Membership.mem (PiNat.cylinder x n) y → Eq (PiNat.cylinder y n) (PiNat.cylin …
    -/
  · intro hy
    /-
      case mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hy : Membership.mem (PiNat.cylinder x n) y
      ⊢ Eq (PiNat.cylinder y n) (PiNat.cylinder x n)
    -/
    apply Subset.antisymm
      /-
        case mp.h₁
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        ⊢ HasSubset.Subset (PiNat.cylinder y n) (PiNat.cylinder x n)
      -/
    · intro z hz i hi
      /-
        case mp.h₁
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        z : (n : Nat) → E n
        hz : Membership.mem (PiNat.cylinder y n) z
        i : Nat
        hi : LT.lt i n
        ⊢ Eq (z i) (x i)
      -/
      rw [← hy i hi]
      /-
        case mp.h₁
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        z : (n : Nat) → E n
        hz : Membership.mem (PiNat.cylinder y n) z
        i : Nat
        hi : LT.lt i n
        ⊢ Eq (z i) (y i)
      -/
      exact hz i hi
      /-
        🎉 no goals
      -/
      /-
        case mp.h₂
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        ⊢ HasSubset.Subset (PiNat.cylinder x n) (PiNat.cylinder y n)
      -/
    · intro z hz i hi
      /-
        case mp.h₂
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        z : (n : Nat) → E n
        hz : Membership.mem (PiNat.cylinder x n) z
        i : Nat
        hi : LT.lt i n
        ⊢ Eq (z i) (y i)
      -/
      rw [hy i hi]
      /-
        case mp.h₂
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        n : Nat
        hy : Membership.mem (PiNat.cylinder x n) y
        z : (n : Nat) → E n
        hz : Membership.mem (PiNat.cylinder x n) z
        i : Nat
        hi : LT.lt i n
        ⊢ Eq (z i) (x i)
      -/
      exact hz i hi
      /-
        🎉 no goals
      -/
    /-
      case mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      ⊢ Eq (PiNat.cylinder y n) (PiNat.cylinder x n) → Membership.mem (PiNat.cylinde …
    -/
  · intro h
    /-
      case mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      h : Eq (PiNat.cylinder y n) (PiNat.cylinder x n)
      ⊢ Membership.mem (PiNat.cylinder x n) y
    -/
    rw [← h]
    /-
      case mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      h : Eq (PiNat.cylinder y n) (PiNat.cylinder x n)
      ⊢ Membership.mem (PiNat.cylinder y n) y
    -/
    exact self_mem_cylinder _ _
    /-
      🎉 no goals
    -/


theorem mem_cylinder_comm (x y : ∀ n, E n) (n : ℕ) : y ∈ cylinder x n ↔ x ∈ cylinder y n := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (Membership.mem (PiNat.cylinder  …
  -/
  simp [mem_cylinder_iff_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem mem_cylinder_iff_le_firstDiff {x y : ∀ n, E n} (hne : x ≠ y) (i : ℕ) :
    x ∈ cylinder y i ↔ i ≤ firstDiff x y := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    hne : Ne x y
    i : Nat
    ⊢ Iff (Membership.mem (PiNat.cylinder y i) x) (LE.le i (PiNat.firstDiff x y))
  -/
  constructor
    /-
      case mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hne : Ne x y
      i : Nat
      ⊢ Membership.mem (PiNat.cylinder y i) x → LE.le i (PiNat.firstDiff x y)
    -/
  · intro h
    /-
      case mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hne : Ne x y
      i : Nat
      h : Membership.mem (PiNat.cylinder y i) x
      ⊢ LE.le i (PiNat.firstDiff x y)
    -/
    by_contra!
    /-
      case mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hne : Ne x y
      i : Nat
      h : Membership.mem (PiNat.cylinder y i) x
      this : LT.lt (PiNat.firstDiff x y) i
      ⊢ False
    -/
    exact apply_firstDiff_ne hne (h _ this)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hne : Ne x y
      i : Nat
      ⊢ LE.le i (PiNat.firstDiff x y) → Membership.mem (PiNat.cylinder y i) x
    -/
  · intro hi j hj
    /-
      case mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hne : Ne x y
      i : Nat
      hi : LE.le i (PiNat.firstDiff x y)
      j : Nat
      hj : LT.lt j i
      ⊢ Eq (x j) (y j)
    -/
    exact apply_eq_of_lt_firstDiff (hj.trans_le hi)
    /-
      🎉 no goals
    -/


theorem mem_cylinder_firstDiff (x y : ∀ n, E n) : x ∈ cylinder y (firstDiff x y) := fun _i hi =>
  apply_eq_of_lt_firstDiff hi


theorem cylinder_eq_cylinder_of_le_firstDiff (x y : ∀ n, E n) {n : ℕ} (hn : n ≤ firstDiff x y) :
    cylinder x n = cylinder y n := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hn : LE.le n (PiNat.firstDiff x y)
    ⊢ Eq (PiNat.cylinder x n) (PiNat.cylinder y n)
  -/
  rw [← mem_cylinder_iff_eq]
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hn : LE.le n (PiNat.firstDiff x y)
    ⊢ Membership.mem (PiNat.cylinder y n) x
  -/
  intro i hi
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hn : LE.le n (PiNat.firstDiff x y)
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (x i) (y i)
  -/
  exact apply_eq_of_lt_firstDiff (hi.trans_le hn)
  /-
    🎉 no goals
  -/


theorem iUnion_cylinder_update (x : ∀ n, E n) (n : ℕ) :
    ⋃ k, cylinder (update x n k) (n + 1) = cylinder x n := by
  /-
    E : Nat → Type u_1
    x : (n : Nat) → E n
    n : Nat
    ⊢ Eq (Set.iUnion fun k => PiNat.cylinder (Function.update x n k) (HAdd.hAdd n  …
  -/
  ext y
  /-
    case h
    E : Nat → Type u_1
    x : (n : Nat) → E n
    n : Nat
    y : (n : Nat) → E n
    ⊢ Iff (Membership.mem (Set.iUnion fun k => PiNat.cylinder (Function.update x n …
  -/
  simp only [mem_cylinder_iff, mem_iUnion]
  /-
    case h
    E : Nat → Type u_1
    x : (n : Nat) → E n
    n : Nat
    y : (n : Nat) → E n
    ⊢ Iff (Exists fun i => ∀ (i_1 : Nat), LT.lt i_1 (HAdd.hAdd n 1) → Eq (y i_1) ( …
  -/
  constructor
    /-
      case h.mp
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n : Nat
      y : (n : Nat) → E n
      ⊢ (Exists fun i => ∀ (i_1 : Nat), LT.lt i_1 (HAdd.hAdd n 1) → Eq (y i_1) (Func …
    -/
  · rintro ⟨k, hk⟩ i hi
    /-
      case h.mp.intro
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n : Nat
      y : (n : Nat) → E n
      k : E n
      hk : ∀ (i : Nat), LT.lt i (HAdd.hAdd n 1) → Eq (y i) (Function.update x n k i)
      i : Nat
      hi : LT.lt i n
      ⊢ Eq (y i) (x i)
    -/
    simpa [hi.ne] using hk i (Nat.lt_succ_of_lt hi)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n : Nat
      y : (n : Nat) → E n
      ⊢ (∀ (i : Nat), LT.lt i n → Eq (y i) (x i)) → Exists fun i => ∀ (i_1 : Nat), L …
    -/
  · intro H
    /-
      case h.mpr
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n : Nat
      y : (n : Nat) → E n
      H : ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
      ⊢ Exists fun i => ∀ (i_1 : Nat), LT.lt i_1 (HAdd.hAdd n 1) → Eq (y i_1) (Funct …
    -/
    refine ⟨y n, fun i hi => ?_⟩
    /-
      case h.mpr
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n : Nat
      y : (n : Nat) → E n
      H : ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      ⊢ Eq (y i) (Function.update x n (y n) i)
    -/
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with (h'i | rfl)
      /-
        case h.mpr.inl
        E : Nat → Type u_1
        x : (n : Nat) → E n
        n : Nat
        y : (n : Nat) → E n
        H : ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        h'i : LT.lt i n
        ⊢ Eq (y i) (Function.update x n (y n) i)
      -/
    · simp [H i h'i, h'i.ne]
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.inr
        E : Nat → Type u_1
        x y : (n : Nat) → E n
        i : Nat
        H : ∀ (i_1 : Nat), LT.lt i_1 i → Eq (y i_1) (x i_1)
        hi : LT.lt i (HAdd.hAdd i 1)
        ⊢ Eq (y i) (Function.update x i (y i) i)
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem update_mem_cylinder (x : ∀ n, E n) (n : ℕ) (y : E n) : update x n y ∈ cylinder x n :=
                                    /-
                                      E : Nat → Type u_1
                                      x : (n : Nat) → E n
                                      n : Nat
                                      y : E n
                                      i : Nat
                                      hi : LT.lt i n
                                      ⊢ Eq (Function.update x n y i) (x i)
                                    -/
  mem_cylinder_iff.2 fun i hi => by simp [hi.ne]
                                    /-
                                      🎉 no goals
                                    -/


/-- In the case where `E` has constant value `α`,
the cylinder `cylinder x n` can be identified with the element of `List α`
consisting of the first `n` entries of `x`. See `cylinder_eq_res`.
We call this list `res x n`, the restriction of `x` to `n`. -/
def res (x : ℕ → α) : ℕ → List α
  | 0 => nil
  | Nat.succ n => x n :: res x n


@[simp]
theorem res_zero (x : ℕ → α) : res x 0 = @nil α :=
  rfl


@[simp]
theorem res_succ (x : ℕ → α) (n : ℕ) : res x n.succ = x n :: res x n :=
  rfl


@[simp]
                                                                    /-
                                                                      α : Type u_2
                                                                      x : Nat → α
                                                                      n : Nat
                                                                      ⊢ Eq (PiNat.res x n).length n
                                                                    -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem res_length (x : ℕ → α) (n : ℕ) : (res x n).length = n := by induction n <;> simp [*]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- The restrictions of `x` and `y` to `n` are equal if and only if `x m = y m` for all `m < n`. -/
theorem res_eq_res {x y : ℕ → α} {n : ℕ} :
    res x n = res y n ↔ ∀ ⦃m⦄, m < n → x m = y m := by
  /-
    α : Type u_2
    x y : Nat → α
    n : Nat
    ⊢ Iff (Eq (PiNat.res x n) (PiNat.res y n)) (∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m)  …
  -/
  constructor <;> intro h <;> induction' n with n ih; · simp
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      case mp.succ
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      h : Eq (PiNat.res x (HAdd.hAdd n 1)) (PiNat.res y (HAdd.hAdd n 1))
      ⊢ ∀ ⦃m : Nat⦄, LT.lt m (HAdd.hAdd n 1) → Eq (x m) (y m)
    -/
  · intro m hm
    /-
      case mp.succ
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      h : Eq (PiNat.res x (HAdd.hAdd n 1)) (PiNat.res y (HAdd.hAdd n 1))
      m : Nat
      hm : LT.lt m (HAdd.hAdd n 1)
      ⊢ Eq (x m) (y m)
    -/
    rw [Nat.lt_succ_iff_lt_or_eq] at hm
    /-
      case mp.succ
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      h : Eq (PiNat.res x (HAdd.hAdd n 1)) (PiNat.res y (HAdd.hAdd n 1))
      m : Nat
      hm : Or (LT.lt m n) (Eq m n)
      ⊢ Eq (x m) (y m)
    -/
    simp only [res_succ, cons.injEq] at h
    /-
      case mp.succ
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      m : Nat
      hm : Or (LT.lt m n) (Eq m n)
      h : And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
      ⊢ Eq (x m) (y m)
    -/
    cases' hm with hm hm
      /-
        case mp.succ.inl
        α : Type u_2
        x y : Nat → α
        n : Nat
        ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
        m : Nat
        h : And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
        hm : LT.lt m n
        ⊢ Eq (x m) (y m)
      -/
    · exact ih h.2 hm
      /-
        🎉 no goals
      -/
    /-
      case mp.succ.inr
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      m : Nat
      h : And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
      hm : Eq m n
      ⊢ Eq (x m) (y m)
    -/
    rw [hm]
    /-
      case mp.succ.inr
      α : Type u_2
      x y : Nat → α
      n : Nat
      ih : Eq (PiNat.res x n) (PiNat.res y n) → ∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y …
      m : Nat
      h : And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
      hm : Eq m n
      ⊢ Eq (x n) (y n)
    -/
    exact h.1
    /-
      🎉 no goals
    -/
    /-
      case mpr.zero
      α : Type u_2
      x y : Nat → α
      h : ∀ ⦃m : Nat⦄, LT.lt m 0 → Eq (x m) (y m)
      ⊢ Eq (PiNat.res x 0) (PiNat.res y 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case mpr.succ
    α : Type u_2
    x y : Nat → α
    n : Nat
    ih : (∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y m)) → Eq (PiNat.res x n) (PiNat.res …
    h : ∀ ⦃m : Nat⦄, LT.lt m (HAdd.hAdd n 1) → Eq (x m) (y m)
    ⊢ Eq (PiNat.res x (HAdd.hAdd n 1)) (PiNat.res y (HAdd.hAdd n 1))
  -/
  simp only [res_succ, cons.injEq]
  /-
    case mpr.succ
    α : Type u_2
    x y : Nat → α
    n : Nat
    ih : (∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y m)) → Eq (PiNat.res x n) (PiNat.res …
    h : ∀ ⦃m : Nat⦄, LT.lt m (HAdd.hAdd n 1) → Eq (x m) (y m)
    ⊢ And (Eq (x n) (y n)) (Eq (PiNat.res x n) (PiNat.res y n))
  -/
  refine ⟨h (Nat.lt_succ_self _), ih fun m hm => ?_⟩
  /-
    case mpr.succ
    α : Type u_2
    x y : Nat → α
    n : Nat
    ih : (∀ ⦃m : Nat⦄, LT.lt m n → Eq (x m) (y m)) → Eq (PiNat.res x n) (PiNat.res …
    h : ∀ ⦃m : Nat⦄, LT.lt m (HAdd.hAdd n 1) → Eq (x m) (y m)
    m : Nat
    hm : LT.lt m n
    ⊢ Eq (x m) (y m)
  -/
  exact h (hm.trans (Nat.lt_succ_self _))
  /-
    🎉 no goals
  -/


theorem res_injective : Injective (@res α) := by
  /-
    α : Type u_2
    ⊢ Function.Injective PiNat.res
  -/
  intro x y h
  /-
    α : Type u_2
    x y : Nat → α
    h : Eq (PiNat.res x) (PiNat.res y)
    ⊢ Eq x y
  -/
  ext n
  /-
    case h
    α : Type u_2
    x y : Nat → α
    h : Eq (PiNat.res x) (PiNat.res y)
    n : Nat
    ⊢ Eq (x n) (y n)
  -/
  apply res_eq_res.mp _ (Nat.lt_succ_self _)
  /-
    α : Type u_2
    x y : Nat → α
    h : Eq (PiNat.res x) (PiNat.res y)
    n : Nat
    ⊢ Eq (PiNat.res x n.succ) (PiNat.res y n.succ)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


/-- `cylinder x n` is equal to the set of sequences `y` with the same restriction to `n` as `x`. -/
theorem cylinder_eq_res (x : ℕ → α) (n : ℕ) :
    cylinder x n = { y | res y n = res x n } := by
  /-
    α : Type u_2
    x : Nat → α
    n : Nat
    ⊢ Eq (PiNat.cylinder x n) (setOf fun y => Eq (PiNat.res y n) (PiNat.res x n))
  -/
  ext y
  /-
    case h
    α : Type u_2
    x : Nat → α
    n : Nat
    y : Nat → α
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (Membership.mem (setOf fun y =>  …
  -/
  dsimp [cylinder]
  /-
    case h
    α : Type u_2
    x : Nat → α
    n : Nat
    y : Nat → α
    ⊢ Iff (∀ (i : Nat), LT.lt i n → Eq (y i) (x i)) (Eq (PiNat.res y n) (PiNat.res …
  -/
  rw [res_eq_res]
  /-
    🎉 no goals
  -/


open Classical in
/-- The distance function on a product space `Π n, E n`, given by `dist x y = (1/2)^n` where `n` is
the first index at which `x` and `y` differ. -/
protected def dist : Dist (∀ n, E n) :=
  ⟨fun x y => if x ≠ y then (1 / 2 : ℝ) ^ firstDiff x y else 0⟩


theorem dist_eq_of_ne {x y : ∀ n, E n} (h : x ≠ y) : dist x y = (1 / 2 : ℝ) ^ firstDiff x y := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    h : Ne x y
    ⊢ Eq (Dist.dist x y) (HPow.hPow (1 / 2) (PiNat.firstDiff x y))
  -/
  simp [dist, h]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  E : Nat → Type u_1
                                                                  x : (n : Nat) → E n
                                                                  ⊢ Eq (Dist.dist x x) 0
                                                                -/
protected theorem dist_self (x : ∀ n, E n) : dist x x = 0 := by simp [dist]
                                                                /-
                                                                  🎉 no goals
                                                                -/


protected theorem dist_comm (x y : ∀ n, E n) : dist x y = dist y x := by
  classical
  simp [dist, @eq_comm _ x y, firstDiff_comm]


protected theorem dist_nonneg (x y : ∀ n, E n) : 0 ≤ dist x y := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    ⊢ LE.le 0 (Dist.dist x y)
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
      E : Nat → Type u_1
      x : (n : Nat) → E n
      ⊢ LE.le 0 (Dist.dist x x)
    -/
  · simp [dist]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      h : Ne x y
      ⊢ LE.le 0 (Dist.dist x y)
    -/
  · simp [dist, h, zero_le_two]
    /-
      🎉 no goals
    -/


theorem dist_triangle_nonarch (x y z : ∀ n, E n) : dist x z ≤ max (dist x y) (dist y z) := by
  /-
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
  -/
  rcases eq_or_ne x z with (rfl | hxz)
    /-
      case inl
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      ⊢ LE.le (Dist.dist x x) (Max.max (Dist.dist x y) (Dist.dist y x))
    -/
  · simp [PiNat.dist_self x, PiNat.dist_nonneg]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    hxz : Ne x z
    ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
  -/
  rcases eq_or_ne x y with (rfl | hxy)
    /-
      case inr.inl
      E : Nat → Type u_1
      x z : (n : Nat) → E n
      hxz : Ne x z
      ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x x) (Dist.dist x z))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Nat → Type u_1
    x y z : (n : Nat) → E n
    hxz : Ne x z
    hxy : Ne x y
    ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
  -/
  rcases eq_or_ne y z with (rfl | hyz)
    /-
      case inr.inr.inl
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      hxy hxz : Ne x y
      ⊢ LE.le (Dist.dist x y) (Max.max (Dist.dist x y) (Dist.dist y y))
    -/
  · simp
    /-
      🎉 no goals
    -/
  simp only [dist_eq_of_ne, hxz, hxy, hyz, inv_le_inv₀, one_div, inv_pow, zero_lt_two, Ne,
    not_false_iff, le_max_iff, pow_le_pow_iff_right₀, one_lt_two, pow_pos,
    min_le_iff.1 (min_firstDiff_le x y z hxz)]


protected theorem dist_triangle (x y z : ∀ n, E n) : dist x z ≤ dist x y + dist y z :=
  calc
    dist x z ≤ max (dist x y) (dist y z) := dist_triangle_nonarch x y z
    _ ≤ dist x y + dist y z := max_le_add_of_nonneg (PiNat.dist_nonneg _ _) (PiNat.dist_nonneg _ _)


protected theorem eq_of_dist_eq_zero (x y : ∀ n, E n) (hxy : dist x y = 0) : x = y := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    hxy : Eq (Dist.dist x y) 0
    ⊢ Eq x y
  -/
  rcases eq_or_ne x y with (rfl | h); · rfl
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    hxy : Eq (Dist.dist x y) 0
    h : Ne x y
    ⊢ Eq x y
  -/
  simp [dist_eq_of_ne h] at hxy
  /-
    🎉 no goals
  -/


theorem mem_cylinder_iff_dist_le {x y : ∀ n, E n} {n : ℕ} :
    y ∈ cylinder x n ↔ dist y x ≤ (1 / 2) ^ n := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (LE.le (Dist.dist y x) (HPow.hPo …
  -/
  rcases eq_or_ne y x with (rfl | hne)
    /-
      case inl
      E : Nat → Type u_1
      y : (n : Nat) → E n
      n : Nat
      ⊢ Iff (Membership.mem (PiNat.cylinder y n) y) (LE.le (Dist.dist y y) (HPow.hPo …
    -/
  · simp [PiNat.dist_self]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hne : Ne y x
    ⊢ Iff (Membership.mem (PiNat.cylinder x n) y) (LE.le (Dist.dist y x) (HPow.hPo …
  -/
  suffices (∀ i : ℕ, i < n → y i = x i) ↔ n ≤ firstDiff y x by simpa [dist_eq_of_ne hne]
  /-
    case inr
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    hne : Ne y x
    ⊢ Iff (∀ (i : Nat), LT.lt i n → Eq (y i) (x i)) (LE.le n (PiNat.firstDiff y x))
  -/
  constructor
    /-
      case inr.mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hne : Ne y x
      ⊢ (∀ (i : Nat), LT.lt i n → Eq (y i) (x i)) → LE.le n (PiNat.firstDiff y x)
    -/
  · intro hy
    /-
      case inr.mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hne : Ne y x
      hy : ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
      ⊢ LE.le n (PiNat.firstDiff y x)
    -/
    by_contra! H
    /-
      case inr.mp
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hne : Ne y x
      hy : ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
      H : LT.lt (PiNat.firstDiff y x) n
      ⊢ False
    -/
    exact apply_firstDiff_ne hne (hy _ H)
    /-
      🎉 no goals
    -/
    /-
      case inr.mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hne : Ne y x
      ⊢ LE.le n (PiNat.firstDiff y x) → ∀ (i : Nat), LT.lt i n → Eq (y i) (x i)
    -/
  · intro h i hi
    /-
      case inr.mpr
      E : Nat → Type u_1
      x y : (n : Nat) → E n
      n : Nat
      hne : Ne y x
      h : LE.le n (PiNat.firstDiff y x)
      i : Nat
      hi : LT.lt i n
      ⊢ Eq (y i) (x i)
    -/
    exact apply_eq_of_lt_firstDiff (hi.trans_le h)
    /-
      🎉 no goals
    -/


theorem apply_eq_of_dist_lt {x y : ∀ n, E n} {n : ℕ} (h : dist x y < (1 / 2) ^ n) {i : ℕ}
    (hi : i ≤ n) : x i = y i := by
  /-
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    h : LT.lt (Dist.dist x y) (HPow.hPow (1 / 2) n)
    i : Nat
    hi : LE.le i n
    ⊢ Eq (x i) (y i)
  -/
  rcases eq_or_ne x y with (rfl | hne)
    /-
      case inl
      E : Nat → Type u_1
      x : (n : Nat) → E n
      n i : Nat
      hi : LE.le i n
      h : LT.lt (Dist.dist x x) (HPow.hPow (1 / 2) n)
      ⊢ Eq (x i) (x i)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  have : n < firstDiff x y := by
    simpa [dist_eq_of_ne hne, inv_lt_inv₀, pow_lt_pow_iff_right₀, one_lt_two] using h
  /-
    case inr
    E : Nat → Type u_1
    x y : (n : Nat) → E n
    n : Nat
    h : LT.lt (Dist.dist x y) (HPow.hPow (1 / 2) n)
    i : Nat
    hi : LE.le i n
    hne : Ne x y
    this : LT.lt n (PiNat.firstDiff x y)
    ⊢ Eq (x i) (y i)
  -/
  exact apply_eq_of_lt_firstDiff (hi.trans_lt this)
  /-
    🎉 no goals
  -/


/-- A function to a pseudo-metric-space is `1`-Lipschitz if and only if points in the same cylinder
of length `n` are sent to points within distance `(1/2)^n`.
Not expressed using `LipschitzWith` as we don't have a metric space structure -/
theorem lipschitz_with_one_iff_forall_dist_image_le_of_mem_cylinder {α : Type*}
    [PseudoMetricSpace α] {f : (∀ n, E n) → α} :
    (∀ x y : ∀ n, E n, dist (f x) (f y) ≤ dist x y) ↔
      ∀ x y n, y ∈ cylinder x n → dist (f x) (f y) ≤ (1 / 2) ^ n := by
  /-
    E : Nat → Type u_1
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    f : ((n : Nat) → E n) → α
    ⊢ Iff (∀ (x y : (n : Nat) → E n), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y …
  -/
  constructor
    /-
      case mp
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      ⊢ (∀ (x y : (n : Nat) → E n), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)) → …
    -/
  · intro H x y n hxy
    /-
      case mp
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
      x y : (n : Nat) → E n
      n : Nat
      hxy : Membership.mem (PiNat.cylinder x n) y
      ⊢ LE.le (Dist.dist (f x) (f y)) (HPow.hPow (1 / 2) n)
    -/
    apply (H x y).trans
    /-
      case mp
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
      x y : (n : Nat) → E n
      n : Nat
      hxy : Membership.mem (PiNat.cylinder x n) y
      ⊢ LE.le (Dist.dist x y) (HPow.hPow (1 / 2) n)
    -/
    rw [PiNat.dist_comm]
    /-
      case mp
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
      x y : (n : Nat) → E n
      n : Nat
      hxy : Membership.mem (PiNat.cylinder x n) y
      ⊢ LE.le (Dist.dist y x) (HPow.hPow (1 / 2) n)
    -/
    exact mem_cylinder_iff_dist_le.1 hxy
    /-
      🎉 no goals
    -/
    /-
      case mpr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      ⊢ (∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y  …
    -/
  · intro H x y
    /-
      case mpr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
      x y : (n : Nat) → E n
      ⊢ LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
    -/
    rcases eq_or_ne x y with (rfl | hne)
      /-
        case mpr.inl
        E : Nat → Type u_1
        α : Type u_2
        inst✝ : PseudoMetricSpace α
        f : ((n : Nat) → E n) → α
        H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
        x : (n : Nat) → E n
        ⊢ LE.le (Dist.dist (f x) (f x)) (Dist.dist x x)
      -/
    · simp [PiNat.dist_nonneg]
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
      x y : (n : Nat) → E n
      hne : Ne x y
      ⊢ LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
    -/
    rw [dist_eq_of_ne hne]
    /-
      case mpr.inr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
      x y : (n : Nat) → E n
      hne : Ne x y
      ⊢ LE.le (Dist.dist (f x) (f y)) (HPow.hPow (1 / 2) (PiNat.firstDiff x y))
    -/
    apply H x y (firstDiff x y)
    /-
      case mpr.inr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
      x y : (n : Nat) → E n
      hne : Ne x y
      ⊢ Membership.mem (PiNat.cylinder x (PiNat.firstDiff x y)) y
    -/
    rw [firstDiff_comm]
    /-
      case mpr.inr
      E : Nat → Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      f : ((n : Nat) → E n) → α
      H : ∀ (x y : (n : Nat) → E n) (n : Nat), Membership.mem (PiNat.cylinder x n) y …
      x y : (n : Nat) → E n
      hne : Ne x y
      ⊢ Membership.mem (PiNat.cylinder x (PiNat.firstDiff y x)) y
    -/
    exact mem_cylinder_firstDiff _ _
    /-
      🎉 no goals
    -/


theorem isOpen_cylinder (x : ∀ n, E n) (n : ℕ) : IsOpen (cylinder x n) := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    x : (n : Nat) → E n
    n : Nat
    ⊢ IsOpen (PiNat.cylinder x n)
  -/
  rw [PiNat.cylinder_eq_pi]
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    x : (n : Nat) → E n
    n : Nat
    ⊢ IsOpen ((↑(Finset.range n)).pi fun i => Singleton.singleton (x i))
  -/
  exact isOpen_set_pi (Finset.range n).finite_toSet fun a _ => isOpen_discrete _
  /-
    🎉 no goals
  -/


theorem isTopologicalBasis_cylinders :
    IsTopologicalBasis { s : Set (∀ n, E n) | ∃ (x : ∀ n, E n) (n : ℕ), s = cylinder x n } := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun s => Exists fun x => Exists f …
  -/
  apply isTopologicalBasis_of_isOpen_of_nhds
    /-
      case h_open
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      ⊢ ∀ (u : Set ((n : Nat) → E n)), Membership.mem (setOf fun s => Exists fun x = …
    -/
  · rintro u ⟨x, n, rfl⟩
    /-
      case h_open.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      n : Nat
      ⊢ IsOpen (PiNat.cylinder x n)
    -/
    apply isOpen_cylinder
    /-
      🎉 no goals
    -/
    /-
      case h_nhds
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      ⊢ ∀ (a : (n : Nat) → E n) (u : Set ((n : Nat) → E n)), Membership.mem u a → Is …
    -/
  · intro x u hx u_open
    obtain ⟨v, ⟨U, F, -, rfl⟩, xU, Uu⟩ :
        ∃ v ∈ { S : Set (∀ i : ℕ, E i) | ∃ (U : ∀ i : ℕ, Set (E i)) (F : Finset ℕ),
          (∀ i : ℕ, i ∈ F → U i ∈ { s : Set (E i) | IsOpen s }) ∧ S = (F : Set ℕ).pi U },
        x ∈ v ∧ v ⊆ u :=
      (isTopologicalBasis_pi fun n : ℕ => isTopologicalBasis_opens).exists_subset_of_mem_open hx
        u_open
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      ⊢ Exists fun v => And (Membership.mem (setOf fun s => Exists fun x => Exists f …
    -/
    rcases Finset.bddAbove F with ⟨n, hn⟩
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      ⊢ Exists fun v => And (Membership.mem (setOf fun s => Exists fun x => Exists f …
    -/
    refine ⟨cylinder x (n + 1), ⟨x, n + 1, rfl⟩, self_mem_cylinder _ _, Subset.trans ?_ Uu⟩
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      ⊢ HasSubset.Subset (PiNat.cylinder x (HAdd.hAdd n 1)) ((↑F).pi U)
    -/
    intro y hy
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      ⊢ Membership.mem ((↑F).pi U) y
    -/
    suffices ∀ i : ℕ, i ∈ F → y i ∈ U i by simpa
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      ⊢ ∀ (i : Nat), Membership.mem F i → Membership.mem (U i) (y i)
    -/
    intro i hi
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      i : Nat
      hi : Membership.mem F i
      ⊢ Membership.mem (U i) (y i)
    -/
    have : y i = x i := mem_cylinder_iff.1 hy i ((hn hi).trans_lt (lt_add_one n))
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      i : Nat
      hi : Membership.mem F i
      this : Eq (y i) (x i)
      ⊢ Membership.mem (U i) (y i)
    -/
    rw [this]
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      xU : Membership.mem ((↑F).pi U) x
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      i : Nat
      hi : Membership.mem F i
      this : Eq (y i) (x i)
      ⊢ Membership.mem (U i) (x i)
    -/
    simp only [Set.mem_pi, Finset.mem_coe] at xU
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      u : Set ((n : Nat) → E n)
      hx : Membership.mem u x
      u_open : IsOpen u
      U : (i : Nat) → Set (E i)
      F : Finset Nat
      Uu : HasSubset.Subset ((↑F).pi U) u
      n : Nat
      hn : Membership.mem (upperBounds ↑F) n
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x (HAdd.hAdd n 1)) y
      i : Nat
      hi : Membership.mem F i
      this : Eq (y i) (x i)
      xU : ∀ (i : Nat), Membership.mem F i → Membership.mem (U i) (x i)
      ⊢ Membership.mem (U i) (x i)
    -/
    exact xU i hi
    /-
      🎉 no goals
    -/


theorem isOpen_iff_dist (s : Set (∀ n, E n)) :
    IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ∀ y, dist x y < ε → y ∈ s := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    ⊢ Iff (IsOpen s) (∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε = …
  -/
  constructor
    /-
      case mp
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      ⊢ IsOpen s → ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And …
    -/
  · intro hs x hx
    obtain ⟨v, ⟨y, n, rfl⟩, h'x, h's⟩ :
        ∃ v ∈ { s | ∃ (x : ∀ n : ℕ, E n) (n : ℕ), s = cylinder x n }, x ∈ v ∧ v ⊆ s :=
      (isTopologicalBasis_cylinders E).exists_subset_of_mem_open hx hs
    /-
      case mp.intro.intro.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      hs : IsOpen s
      x : (n : Nat) → E n
      hx : Membership.mem s x
      y : (n : Nat) → E n
      n : Nat
      h'x : Membership.mem (PiNat.cylinder y n) x
      h's : HasSubset.Subset (PiNat.cylinder y n) s
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : (n : Nat) → E n), LT.lt (Dist.dist x …
    -/
    rw [← mem_cylinder_iff_eq.1 h'x] at h's
    exact
      ⟨(1 / 2 : ℝ) ^ n, by simp, fun y hy => h's fun i hi => (apply_eq_of_dist_lt hy hi.le).symm⟩
    /-
      case mpr
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      ⊢ (∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε  …
    -/
  · intro h
    /-
      case mpr
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      ⊢ IsOpen s
    -/
    refine (isTopologicalBasis_cylinders E).isOpen_iff.2 fun x hx => ?_
    /-
      case mpr
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ⊢ Exists fun t => And (Membership.mem (setOf fun s => Exists fun x => Exists f …
    -/
    rcases h x hx with ⟨ε, εpos, hε⟩
    /-
      case mpr.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : (n : Nat) → E n), LT.lt (Dist.dist x y) ε → Membership.mem s y
      ⊢ Exists fun t => And (Membership.mem (setOf fun s => Exists fun x => Exists f …
    -/
    obtain ⟨n, hn⟩ : ∃ n : ℕ, (1 / 2 : ℝ) ^ n < ε := exists_pow_lt_of_lt_one εpos one_half_lt_one
    /-
      case mpr.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : (n : Nat) → E n), LT.lt (Dist.dist x y) ε → Membership.mem s y
      n : Nat
      hn : LT.lt (HPow.hPow (1 / 2) n) ε
      ⊢ Exists fun t => And (Membership.mem (setOf fun s => Exists fun x => Exists f …
    -/
    refine ⟨cylinder x n, ⟨x, n, rfl⟩, self_mem_cylinder x n, fun y hy => hε y ?_⟩
    /-
      case mpr.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : (n : Nat) → E n), LT.lt (Dist.dist x y) ε → Membership.mem s y
      n : Nat
      hn : LT.lt (HPow.hPow (1 / 2) n) ε
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x n) y
      ⊢ LT.lt (Dist.dist x y) ε
    -/
    rw [PiNat.dist_comm]
    /-
      case mpr.intro.intro.intro
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      h : ∀ (x : (n : Nat) → E n), Membership.mem s x → Exists fun ε => And (GT.gt ε …
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (y : (n : Nat) → E n), LT.lt (Dist.dist x y) ε → Membership.mem s y
      n : Nat
      hn : LT.lt (HPow.hPow (1 / 2) n) ε
      y : (n : Nat) → E n
      hy : Membership.mem (PiNat.cylinder x n) y
      ⊢ LT.lt (Dist.dist y x) ε
    -/
    exact (mem_cylinder_iff_dist_le.1 hy).trans_lt hn
    /-
      🎉 no goals
    -/


/-- Metric space structure on `Π (n : ℕ), E n` when the spaces `E n` have the discrete topology,
where the distance is given by `dist x y = (1/2)^n`, where `n` is the smallest index where `x` and
`y` differ. Not registered as a global instance by default.
Warning: this definition makes sure that the topology is defeq to the original product topology,
but it does not take care of a possible uniformity. If the `E n` have a uniform structure, then
there will be two non-defeq uniform structures on `Π n, E n`, the product one and the one coming
from the metric structure. In this case, use `metricSpaceOfDiscreteUniformity` instead. -/
protected def metricSpace : MetricSpace (∀ n, E n) :=
  MetricSpace.ofDistTopology dist PiNat.dist_self PiNat.dist_comm PiNat.dist_triangle
    isOpen_iff_dist PiNat.eq_of_dist_eq_zero


/-- Metric space structure on `Π (n : ℕ), E n` when the spaces `E n` have the discrete uniformity,
where the distance is given by `dist x y = (1/2)^n`, where `n` is the smallest index where `x` and
`y` differ. Not registered as a global instance by default. -/
protected def metricSpaceOfDiscreteUniformity {E : ℕ → Type*} [∀ n, UniformSpace (E n)]
    (h : ∀ n, uniformity (E n) = 𝓟 idRel) : MetricSpace (∀ n, E n) :=
  haveI : ∀ n, DiscreteTopology (E n) := fun n => discreteTopology_of_discrete_uniformity (h n)
  { dist_triangle := PiNat.dist_triangle
    dist_comm := PiNat.dist_comm
    dist_self := PiNat.dist_self
    eq_of_dist_eq_zero := PiNat.eq_of_dist_eq_zero _ _
    toUniformSpace := Pi.uniformSpace _
    uniformity_dist := by
      /-
        E✝ : Nat → Type u_1
        inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
        inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
        E : Nat → Type u_2
        inst✝ : (n : Nat) → UniformSpace (E n)
        h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
        this : ∀ (n : Nat), DiscreteTopology (E n)
        ⊢ Eq (uniformity ((n : Nat) → E n)) (iInf fun ε => iInf fun h => Filter.princi …
      -/
      simp only [Pi.uniformity, h, idRel, comap_principal, preimage_setOf_eq]
      /-
        E✝ : Nat → Type u_1
        inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
        inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
        E : Nat → Type u_2
        inst✝ : (n : Nat) → UniformSpace (E n)
        h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
        this : ∀ (n : Nat), DiscreteTopology (E n)
        ⊢ Eq (iInf fun i => Filter.principal (setOf fun a => Eq (a.1 i) (a.2 i))) (iIn …
      -/
      apply le_antisymm
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          ⊢ LE.le (iInf fun i => Filter.principal (setOf fun a => Eq (a.1 i) (a.2 i))) ( …
        -/
      · simp only [le_iInf_iff, le_principal_iff]
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          ⊢ ∀ (i : Real), GT.gt i 0 → Membership.mem (iInf fun i => Filter.principal (se …
        -/
        intro ε εpos
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          ε : Real
          εpos : GT.gt ε 0
          ⊢ Membership.mem (iInf fun i => Filter.principal (setOf fun a => Eq (a.1 i) (a …
        -/
        obtain ⟨n, hn⟩ : ∃ n, (1 / 2 : ℝ) ^ n < ε := exists_pow_lt_of_lt_one εpos (by norm_num)
        apply
          @mem_iInf_of_iInter _ _ _ _ _ (Finset.range n).finite_toSet fun i =>
            { p : (∀ n : ℕ, E n) × ∀ n : ℕ, E n | p.fst i = p.snd i }
          /-
            case a.intro.hV
            E✝ : Nat → Type u_1
            inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
            inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
            E : Nat → Type u_2
            inst✝ : (n : Nat) → UniformSpace (E n)
            h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
            this : ∀ (n : Nat), DiscreteTopology (E n)
            ε : Real
            εpos : GT.gt ε 0
            n : Nat
            hn : LT.lt (HPow.hPow (1 / 2) n) ε
            ⊢ ∀ (i : ↑↑(Finset.range n)), Membership.mem (Filter.principal (setOf fun a => …
          -/
        · simp only [mem_principal, setOf_subset_setOf, imp_self, imp_true_iff]
          /-
            🎉 no goals
          -/
          /-
            case a.intro.hU
            E✝ : Nat → Type u_1
            inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
            inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
            E : Nat → Type u_2
            inst✝ : (n : Nat) → UniformSpace (E n)
            h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
            this : ∀ (n : Nat), DiscreteTopology (E n)
            ε : Real
            εpos : GT.gt ε 0
            n : Nat
            hn : LT.lt (HPow.hPow (1 / 2) n) ε
            ⊢ HasSubset.Subset (Set.iInter fun i => setOf fun p => Eq (p.1 ↑i) (p.2 ↑i)) ( …
          -/
        · rintro ⟨x, y⟩ hxy
          simp only [Finset.mem_coe, Finset.mem_range, iInter_coe_set, mem_iInter, mem_setOf_eq]
            at hxy
          /-
            case a.intro.hU.mk
            E✝ : Nat → Type u_1
            inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
            inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
            E : Nat → Type u_2
            inst✝ : (n : Nat) → UniformSpace (E n)
            h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
            this : ∀ (n : Nat), DiscreteTopology (E n)
            ε : Real
            εpos : GT.gt ε 0
            n : Nat
            hn : LT.lt (HPow.hPow (1 / 2) n) ε
            x y : (n : Nat) → E n
            hxy : ∀ (i : Nat), LT.lt i n → Eq (x i) (y i)
            ⊢ Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { fst := x, snd  …
          -/
          apply lt_of_le_of_lt _ hn
          /-
            E✝ : Nat → Type u_1
            inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
            inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
            E : Nat → Type u_2
            inst✝ : (n : Nat) → UniformSpace (E n)
            h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
            this : ∀ (n : Nat), DiscreteTopology (E n)
            ε : Real
            εpos : GT.gt ε 0
            n : Nat
            hn : LT.lt (HPow.hPow (1 / 2) n) ε
            x y : (n : Nat) → E n
            hxy : ∀ (i : Nat), LT.lt i n → Eq (x i) (y i)
            ⊢ LE.le (Dist.dist { fst := x, snd := y }.1 { fst := x, snd := y }.2) (HPow.hP …
          -/
          rw [← mem_cylinder_iff_dist_le, mem_cylinder_iff]
          /-
            E✝ : Nat → Type u_1
            inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
            inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
            E : Nat → Type u_2
            inst✝ : (n : Nat) → UniformSpace (E n)
            h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
            this : ∀ (n : Nat), DiscreteTopology (E n)
            ε : Real
            εpos : GT.gt ε 0
            n : Nat
            hn : LT.lt (HPow.hPow (1 / 2) n) ε
            x y : (n : Nat) → E n
            hxy : ∀ (i : Nat), LT.lt i n → Eq (x i) (y i)
            ⊢ ∀ (i : Nat), LT.lt i n → Eq ({ fst := x, snd := y }.1 i) ({ fst := x, snd := …
          -/
          exact hxy
          /-
            🎉 no goals
          -/
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          ⊢ LE.le (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => LT.lt (D …
        -/
      · simp only [le_iInf_iff, le_principal_iff]
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          ⊢ ∀ (i : Nat), Membership.mem (iInf fun ε => iInf fun h => Filter.principal (s …
        -/
        intro n
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          n : Nat
          ⊢ Membership.mem (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => …
        -/
        refine mem_iInf_of_mem ((1 / 2) ^ n : ℝ) ?_
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          n : Nat
          ⊢ Membership.mem (iInf fun h => Filter.principal (setOf fun p => LT.lt (Dist.d …
        -/
        refine mem_iInf_of_mem (by positivity) ?_
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          n : Nat
          ⊢ Membership.mem (Filter.principal (setOf fun p => LT.lt (Dist.dist p.1 p.2) ( …
        -/
        simp only [mem_principal, setOf_subset_setOf, Prod.forall]
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          n : Nat
          ⊢ ∀ (a b : (n : Nat) → E n), LT.lt (Dist.dist a b) (HPow.hPow (1 / 2) n) → Eq  …
        -/
        intro x y hxy
        /-
          case a
          E✝ : Nat → Type u_1
          inst✝² : (n : Nat) → TopologicalSpace (E✝ n)
          inst✝¹ : ∀ (n : Nat), DiscreteTopology (E✝ n)
          E : Nat → Type u_2
          inst✝ : (n : Nat) → UniformSpace (E n)
          h : ∀ (n : Nat), Eq (uniformity (E n)) (Filter.principal idRel)
          this : ∀ (n : Nat), DiscreteTopology (E n)
          n : Nat
          x y : (n : Nat) → E n
          hxy : LT.lt (Dist.dist x y) (HPow.hPow (1 / 2) n)
          ⊢ Eq (x n) (y n)
        -/
        exact apply_eq_of_dist_lt hxy le_rfl }
        /-
          🎉 no goals
        -/


/-- Metric space structure on `ℕ → ℕ` where the distance is given by `dist x y = (1/2)^n`,
where `n` is the smallest index where `x` and `y` differ.
Not registered as a global instance by default. -/
def metricSpaceNatNat : MetricSpace (ℕ → ℕ) :=
  PiNat.metricSpaceOfDiscreteUniformity fun _ => rfl


protected theorem completeSpace : CompleteSpace (∀ n, E n) := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    ⊢ CompleteSpace ((n : Nat) → E n)
  -/
  refine Metric.complete_of_convergent_controlled_sequences (fun n => (1 / 2) ^ n) (by simp) ?_
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    ⊢ ∀ (u : Nat → (n : Nat) → E n), (∀ (N n m : Nat), LE.le N n → LE.le N m → LT. …
  -/
  intro u hu
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    u : Nat → (n : Nat) → E n
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  refine ⟨fun n => u n n, tendsto_pi_nhds.2 fun i => ?_⟩
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    u : Nat → (n : Nat) → E n
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    i : Nat
    ⊢ Filter.Tendsto (fun i_1 => u i_1 i) Filter.atTop (nhds (u i i))
  -/
  refine tendsto_const_nhds.congr' ?_
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    u : Nat → (n : Nat) → E n
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    i : Nat
    ⊢ Filter.atTop.EventuallyEq (fun x => u i i) fun i_1 => u i_1 i
  -/
  filter_upwards [Filter.Ici_mem_atTop i] with n hn
  /-
    case h
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    u : Nat → (n : Nat) → E n
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    i n : Nat
    hn : Membership.mem (Set.Ici i) n
    ⊢ Eq (u i i) (u n i)
  -/
  exact apply_eq_of_dist_lt (hu i i n le_rfl hn) le_rfl
  /-
    🎉 no goals
  -/


theorem exists_disjoint_cylinder {s : Set (∀ n, E n)} (hs : IsClosed s) {x : ∀ n, E n}
    (hx : x ∉ s) : ∃ n, Disjoint s (cylinder x n) := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    ⊢ Exists fun n => Disjoint s (PiNat.cylinder x n)
  -/
  rcases eq_empty_or_nonempty s with (rfl | hne)
    /-
      case inl
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      x : (n : Nat) → E n
      hs : IsClosed EmptyCollection.emptyCollection
      hx : Not (Membership.mem EmptyCollection.emptyCollection x)
      ⊢ Exists fun n => Disjoint EmptyCollection.emptyCollection (PiNat.cylinder x n)
    -/
  · exact ⟨0, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hne : s.Nonempty
    ⊢ Exists fun n => Disjoint s (PiNat.cylinder x n)
  -/
  have A : 0 < infDist x s := (hs.not_mem_iff_infDist_pos hne).1 hx
  /-
    case inr
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hne : s.Nonempty
    A : LT.lt 0 (Metric.infDist x s)
    ⊢ Exists fun n => Disjoint s (PiNat.cylinder x n)
  -/
  obtain ⟨n, hn⟩ : ∃ n, (1 / 2 : ℝ) ^ n < infDist x s := exists_pow_lt_of_lt_one A one_half_lt_one
  /-
    case inr.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hne : s.Nonempty
    A : LT.lt 0 (Metric.infDist x s)
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) (Metric.infDist x s)
    ⊢ Exists fun n => Disjoint s (PiNat.cylinder x n)
  -/
  refine ⟨n, disjoint_left.2 fun y ys hy => ?_⟩
  /-
    case inr.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hne : s.Nonempty
    A : LT.lt 0 (Metric.infDist x s)
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) (Metric.infDist x s)
    y : (n : Nat) → E n
    ys : Membership.mem s y
    hy : Membership.mem (PiNat.cylinder x n) y
    ⊢ False
  -/
  apply lt_irrefl (infDist x s)
  calc
    infDist x s ≤ dist x y := infDist_le_dist_of_mem ys
    _ ≤ (1 / 2) ^ n := by
      rw [mem_cylinder_comm] at hy
      exact mem_cylinder_iff_dist_le.1 hy
    _ < infDist x s := hn


open Classical in
/-- Given a point `x` in a product space `Π (n : ℕ), E n`, and `s` a subset of this space, then
`shortestPrefixDiff x s` if the smallest `n` for which there is no element of `s` having the same
prefix of length `n` as `x`. If there is no such `n`, then use `0` by convention. -/
def shortestPrefixDiff {E : ℕ → Type*} (x : ∀ n, E n) (s : Set (∀ n, E n)) : ℕ :=
  if h : ∃ n, Disjoint s (cylinder x n) then Nat.find h else 0


theorem firstDiff_lt_shortestPrefixDiff {s : Set (∀ n, E n)} (hs : IsClosed s) {x y : ∀ n, E n}
    (hx : x ∉ s) (hy : y ∈ s) : firstDiff x y < shortestPrefixDiff x s := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x y : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hy : Membership.mem s y
    ⊢ LT.lt (PiNat.firstDiff x y) (PiNat.shortestPrefixDiff x s)
  -/
  have A := exists_disjoint_cylinder hs hx
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x y : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hy : Membership.mem s y
    A : Exists fun n => Disjoint s (PiNat.cylinder x n)
    ⊢ LT.lt (PiNat.firstDiff x y) (PiNat.shortestPrefixDiff x s)
  -/
  rw [shortestPrefixDiff, dif_pos A]
  classical
  have B := Nat.find_spec A
  contrapose! B
  rw [not_disjoint_iff_nonempty_inter]
  refine ⟨y, hy, ?_⟩
  rw [mem_cylinder_comm]
  exact cylinder_anti y B (mem_cylinder_firstDiff x y)


theorem shortestPrefixDiff_pos {s : Set (∀ n, E n)} (hs : IsClosed s) (hne : s.Nonempty)
    {x : ∀ n, E n} (hx : x ∉ s) : 0 < shortestPrefixDiff x s := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    ⊢ LT.lt 0 (PiNat.shortestPrefixDiff x s)
  -/
  rcases hne with ⟨y, hy⟩
  /-
    case intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    y : (n : Nat) → E n
    hy : Membership.mem s y
    ⊢ LT.lt 0 (PiNat.shortestPrefixDiff x s)
  -/
  exact (zero_le _).trans_lt (firstDiff_lt_shortestPrefixDiff hs hx hy)
  /-
    🎉 no goals
  -/


/-- Given a point `x` in a product space `Π (n : ℕ), E n`, and `s` a subset of this space, then
`longestPrefix x s` if the largest `n` for which there is an element of `s` having the same
prefix of length `n` as `x`. If there is no such `n`, use `0` by convention. -/
def longestPrefix {E : ℕ → Type*} (x : ∀ n, E n) (s : Set (∀ n, E n)) : ℕ :=
  shortestPrefixDiff x s - 1


theorem firstDiff_le_longestPrefix {s : Set (∀ n, E n)} (hs : IsClosed s) {x y : ∀ n, E n}
    (hx : x ∉ s) (hy : y ∈ s) : firstDiff x y ≤ longestPrefix x s := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x y : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    hy : Membership.mem s y
    ⊢ LE.le (PiNat.firstDiff x y) (PiNat.longestPrefix x s)
  -/
  rw [longestPrefix, le_tsub_iff_right]
    /-
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      hs : IsClosed s
      x y : (n : Nat) → E n
      hx : Not (Membership.mem s x)
      hy : Membership.mem s y
      ⊢ LE.le (HAdd.hAdd (PiNat.firstDiff x y) 1) (PiNat.shortestPrefixDiff x s)
    -/
  · exact firstDiff_lt_shortestPrefixDiff hs hx hy
    /-
      🎉 no goals
    -/
    /-
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      hs : IsClosed s
      x y : (n : Nat) → E n
      hx : Not (Membership.mem s x)
      hy : Membership.mem s y
      ⊢ LE.le 1 (PiNat.shortestPrefixDiff x s)
    -/
  · exact shortestPrefixDiff_pos hs ⟨y, hy⟩ hx
    /-
      🎉 no goals
    -/


theorem inter_cylinder_longestPrefix_nonempty {s : Set (∀ n, E n)} (hs : IsClosed s)
    (hne : s.Nonempty) (x : ∀ n, E n) : (s ∩ cylinder x (longestPrefix x s)).Nonempty := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    x : (n : Nat) → E n
    ⊢ (Inter.inter s (PiNat.cylinder x (PiNat.longestPrefix x s))).Nonempty
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      E : Nat → Type u_1
      inst✝¹ : (n : Nat) → TopologicalSpace (E n)
      inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
      s : Set ((n : Nat) → E n)
      hs : IsClosed s
      hne : s.Nonempty
      x : (n : Nat) → E n
      hx : Membership.mem s x
      ⊢ (Inter.inter s (PiNat.cylinder x (PiNat.longestPrefix x s))).Nonempty
    -/
  · exact ⟨x, hx, self_mem_cylinder _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    ⊢ (Inter.inter s (PiNat.cylinder x (PiNat.longestPrefix x s))).Nonempty
  -/
  have A := exists_disjoint_cylinder hs hx
  have B : longestPrefix x s < shortestPrefixDiff x s :=
    Nat.pred_lt (shortestPrefixDiff_pos hs hne hx).ne'
  /-
    case neg
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    A : Exists fun n => Disjoint s (PiNat.cylinder x n)
    B : LT.lt (PiNat.longestPrefix x s) (PiNat.shortestPrefixDiff x s)
    ⊢ (Inter.inter s (PiNat.cylinder x (PiNat.longestPrefix x s))).Nonempty
  -/
  rw [longestPrefix, shortestPrefixDiff, dif_pos A] at B ⊢
  classical
  obtain ⟨y, ys, hy⟩ : ∃ y : ∀ n : ℕ, E n, y ∈ s ∧ x ∈ cylinder y (Nat.find A - 1) := by
    simpa only [not_disjoint_iff, mem_cylinder_comm] using Nat.find_min A B
  refine ⟨y, ys, ?_⟩
  rw [mem_cylinder_iff_eq] at hy ⊢
  rw [hy]


theorem disjoint_cylinder_of_longestPrefix_lt {s : Set (∀ n, E n)} (hs : IsClosed s) {x : ∀ n, E n}
    (hx : x ∉ s) {n : ℕ} (hn : longestPrefix x s < n) : Disjoint s (cylinder x n) := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    n : Nat
    hn : LT.lt (PiNat.longestPrefix x s) n
    ⊢ Disjoint s (PiNat.cylinder x n)
  -/
  contrapose! hn
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    n : Nat
    hn : Not (Disjoint s (PiNat.cylinder x n))
    ⊢ LE.le n (PiNat.longestPrefix x s)
  -/
  rcases not_disjoint_iff_nonempty_inter.1 hn with ⟨y, ys, hy⟩
  /-
    case intro.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    n : Nat
    hn : Not (Disjoint s (PiNat.cylinder x n))
    y : (n : Nat) → E n
    ys : Membership.mem s y
    hy : Membership.mem (PiNat.cylinder x n) y
    ⊢ LE.le n (PiNat.longestPrefix x s)
  -/
  apply le_trans _ (firstDiff_le_longestPrefix hs hx ys)
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    n : Nat
    hn : Not (Disjoint s (PiNat.cylinder x n))
    y : (n : Nat) → E n
    ys : Membership.mem s y
    hy : Membership.mem (PiNat.cylinder x n) y
    ⊢ LE.le n (PiNat.firstDiff x y)
  -/
  apply (mem_cylinder_iff_le_firstDiff (ne_of_mem_of_not_mem ys hx).symm _).1
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    x : (n : Nat) → E n
    hx : Not (Membership.mem s x)
    n : Nat
    hn : Not (Disjoint s (PiNat.cylinder x n))
    y : (n : Nat) → E n
    ys : Membership.mem s y
    hy : Membership.mem (PiNat.cylinder x n) y
    ⊢ Membership.mem (PiNat.cylinder y n) x
  -/
  rwa [mem_cylinder_comm]
  /-
    🎉 no goals
  -/


/-- If two points `x, y` coincide up to length `n`, and the longest common prefix of `x` with `s`
is strictly shorter than `n`, then the longest common prefix of `y` with `s` is the same, and both
cylinders of this length based at `x` and `y` coincide. -/
theorem cylinder_longestPrefix_eq_of_longestPrefix_lt_firstDiff {x y : ∀ n, E n}
    {s : Set (∀ n, E n)} (hs : IsClosed s) (hne : s.Nonempty)
    (H : longestPrefix x s < firstDiff x y) (xs : x ∉ s) (ys : y ∉ s) :
    cylinder x (longestPrefix x s) = cylinder y (longestPrefix y s) := by
  have l_eq : longestPrefix y s = longestPrefix x s := by
    rcases lt_trichotomy (longestPrefix y s) (longestPrefix x s) with (L | L | L)
    · have Ax : (s ∩ cylinder x (longestPrefix x s)).Nonempty :=
        inter_cylinder_longestPrefix_nonempty hs hne x
      have Z := disjoint_cylinder_of_longestPrefix_lt hs ys L
      rw [firstDiff_comm] at H
      rw [cylinder_eq_cylinder_of_le_firstDiff _ _ H.le] at Z
      exact (Ax.not_disjoint Z).elim
    · exact L
    · have Ay : (s ∩ cylinder y (longestPrefix y s)).Nonempty :=
        inter_cylinder_longestPrefix_nonempty hs hne y
      have A'y : (s ∩ cylinder y (longestPrefix x s).succ).Nonempty :=
        Ay.mono (inter_subset_inter_right s (cylinder_anti _ L))
      have Z := disjoint_cylinder_of_longestPrefix_lt hs xs (Nat.lt_succ_self _)
      rw [cylinder_eq_cylinder_of_le_firstDiff _ _ H] at Z
      exact (A'y.not_disjoint Z).elim
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    x y : (n : Nat) → E n
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    H : LT.lt (PiNat.longestPrefix x s) (PiNat.firstDiff x y)
    xs : Not (Membership.mem s x)
    ys : Not (Membership.mem s y)
    l_eq : Eq (PiNat.longestPrefix y s) (PiNat.longestPrefix x s)
    ⊢ Eq (PiNat.cylinder x (PiNat.longestPrefix x s)) (PiNat.cylinder y (PiNat.lon …
  -/
  rw [l_eq, ← mem_cylinder_iff_eq]
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    x y : (n : Nat) → E n
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    H : LT.lt (PiNat.longestPrefix x s) (PiNat.firstDiff x y)
    xs : Not (Membership.mem s x)
    ys : Not (Membership.mem s y)
    l_eq : Eq (PiNat.longestPrefix y s) (PiNat.longestPrefix x s)
    ⊢ Membership.mem (PiNat.cylinder y (PiNat.longestPrefix x s)) x
  -/
  exact cylinder_anti y H.le (mem_cylinder_firstDiff x y)
  /-
    🎉 no goals
  -/


/-- Given a closed nonempty subset `s` of `Π (n : ℕ), E n`, there exists a Lipschitz retraction
onto this set, i.e., a Lipschitz map with range equal to `s`, equal to the identity on `s`. -/
theorem exists_lipschitz_retraction_of_isClosed {s : Set (∀ n, E n)} (hs : IsClosed s)
    (hne : s.Nonempty) :
    ∃ f : (∀ n, E n) → ∀ n, E n, (∀ x ∈ s, f x = x) ∧ range f = s ∧ LipschitzWith 1 f := by
  /- The map `f` is defined as follows. For `x ∈ s`, let `f x = x`. Otherwise, consider the longest
    prefix `w` that `x` shares with an element of `s`, and let `f x = z_w` where `z_w` is an element
    of `s` starting with `w`. All the desired properties are clear, except the fact that `f` is
    `1`-Lipschitz: if two points `x, y` belong to a common cylinder of length `n`, one should show
    that their images also belong to a common cylinder of length `n`. This is a case analysis:
    * if both `x, y ∈ s`, then this is clear.
    * if `x ∈ s` but `y ∉ s`, then the longest prefix `w` of `y` shared by an element of `s` is of
    length at least `n` (because of `x`), and then `f y` starts with `w` and therefore stays in the
    same length `n` cylinder.
    * if `x ∉ s`, `y ∉ s`, let `w` be the longest prefix of `x` shared by an element of `s`. If its
    length is `< n`, then it is also the longest prefix of `y`, and we get `f x = f y = z_w`.
    Otherwise, `f x` remains in the same `n`-cylinder as `x`. Similarly for `y`. Finally, `f x` and
    `f y` are again in the same `n`-cylinder, as desired. -/
  classical
  set f := fun x => if x ∈ s then x else (inter_cylinder_longestPrefix_nonempty hs hne x).some
  have fs : ∀ x ∈ s, f x = x := fun x xs => by simp [f, xs]
  refine ⟨f, fs, ?_, ?_⟩
  -- check that the range of `f` is `s`.
  · apply Subset.antisymm
    · rintro x ⟨y, rfl⟩
      by_cases hy : y ∈ s
      · rwa [fs y hy]
      simpa [f, if_neg hy] using (inter_cylinder_longestPrefix_nonempty hs hne y).choose_spec.1
    · intro x hx
      rw [← fs x hx]
      exact mem_range_self _
  -- check that `f` is `1`-Lipschitz, by a case analysis.
  · refine LipschitzWith.mk_one fun x y => ?_
    -- exclude the trivial cases where `x = y`, or `f x = f y`.
    rcases eq_or_ne x y with (rfl | hxy)
    · simp
    rcases eq_or_ne (f x) (f y) with (h' | hfxfy)
    · simp [h', dist_nonneg]
    have I2 : cylinder x (firstDiff x y) = cylinder y (firstDiff x y) := by
      rw [← mem_cylinder_iff_eq]
      apply mem_cylinder_firstDiff
    suffices firstDiff x y ≤ firstDiff (f x) (f y) by
      simpa [dist_eq_of_ne hxy, dist_eq_of_ne hfxfy]
    -- case where `x ∈ s`
    by_cases xs : x ∈ s
    · rw [fs x xs] at hfxfy ⊢
      -- case where `y ∈ s`, trivial
      by_cases ys : y ∈ s
      · rw [fs y ys]
      -- case where `y ∉ s`
      have A : (s ∩ cylinder y (longestPrefix y s)).Nonempty :=
        inter_cylinder_longestPrefix_nonempty hs hne y
      have fy : f y = A.some := by simp_rw [f, if_neg ys]
      have I : cylinder A.some (firstDiff x y) = cylinder y (firstDiff x y) := by
        rw [← mem_cylinder_iff_eq, firstDiff_comm]
        apply cylinder_anti y _ A.some_mem.2
        exact firstDiff_le_longestPrefix hs ys xs
      rwa [← fy, ← I2, ← mem_cylinder_iff_eq, mem_cylinder_iff_le_firstDiff hfxfy.symm,
        firstDiff_comm _ x] at I
    -- case where `x ∉ s`
    · by_cases ys : y ∈ s
      -- case where `y ∈ s` (similar to the above)
      · have A : (s ∩ cylinder x (longestPrefix x s)).Nonempty :=
          inter_cylinder_longestPrefix_nonempty hs hne x
        have fx : f x = A.some := by simp_rw [f, if_neg xs]
        have I : cylinder A.some (firstDiff x y) = cylinder x (firstDiff x y) := by
          rw [← mem_cylinder_iff_eq]
          apply cylinder_anti x _ A.some_mem.2
          apply firstDiff_le_longestPrefix hs xs ys
        rw [fs y ys] at hfxfy ⊢
        rwa [← fx, I2, ← mem_cylinder_iff_eq, mem_cylinder_iff_le_firstDiff hfxfy] at I
      -- case where `y ∉ s`
      · have Ax : (s ∩ cylinder x (longestPrefix x s)).Nonempty :=
          inter_cylinder_longestPrefix_nonempty hs hne x
        have fx : f x = Ax.some := by simp_rw [f, if_neg xs]
        have Ay : (s ∩ cylinder y (longestPrefix y s)).Nonempty :=
          inter_cylinder_longestPrefix_nonempty hs hne y
        have fy : f y = Ay.some := by simp_rw [f, if_neg ys]
        -- case where the common prefix to `x` and `s`, or `y` and `s`, is shorter than the
        -- common part to `x` and `y` -- then `f x = f y`.
        by_cases H : longestPrefix x s < firstDiff x y ∨ longestPrefix y s < firstDiff x y
        · have : cylinder x (longestPrefix x s) = cylinder y (longestPrefix y s) := by
            cases' H with H H
            · exact cylinder_longestPrefix_eq_of_longestPrefix_lt_firstDiff hs hne H xs ys
            · symm
              rw [firstDiff_comm] at H
              exact cylinder_longestPrefix_eq_of_longestPrefix_lt_firstDiff hs hne H ys xs
          rw [fx, fy] at hfxfy
          apply (hfxfy _).elim
          congr
        -- case where the common prefix to `x` and `s` is long, as well as the common prefix to
        -- `y` and `s`. Then all points remain in the same cylinders.
        · push_neg at H
          have I1 : cylinder Ax.some (firstDiff x y) = cylinder x (firstDiff x y) := by
            rw [← mem_cylinder_iff_eq]
            exact cylinder_anti x H.1 Ax.some_mem.2
          have I3 : cylinder y (firstDiff x y) = cylinder Ay.some (firstDiff x y) := by
            rw [eq_comm, ← mem_cylinder_iff_eq]
            exact cylinder_anti y H.2 Ay.some_mem.2
          have : cylinder Ax.some (firstDiff x y) = cylinder Ay.some (firstDiff x y) := by
            rw [I1, I2, I3]
          rw [← fx, ← fy, ← mem_cylinder_iff_eq, mem_cylinder_iff_le_firstDiff hfxfy] at this
          exact this


/-- Given a closed nonempty subset `s` of `Π (n : ℕ), E n`, there exists a retraction onto this
set, i.e., a continuous map with range equal to `s`, equal to the identity on `s`. -/
theorem exists_retraction_of_isClosed {s : Set (∀ n, E n)} (hs : IsClosed s) (hne : s.Nonempty) :
    ∃ f : (∀ n, E n) → ∀ n, E n, (∀ x ∈ s, f x = x) ∧ range f = s ∧ Continuous f := by
  /-
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    ⊢ Exists fun f => And (∀ (x : (n : Nat) → E n), Membership.mem s x → Eq (f x)  …
  -/
  rcases exists_lipschitz_retraction_of_isClosed hs hne with ⟨f, fs, frange, hf⟩
  /-
    case intro.intro.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    s : Set ((n : Nat) → E n)
    hs : IsClosed s
    hne : s.Nonempty
    f : ((n : Nat) → E n) → (n : Nat) → E n
    fs : ∀ (x : (n : Nat) → E n), Membership.mem s x → Eq (f x) x
    frange : Eq (Set.range f) s
    hf : LipschitzWith 1 f
    ⊢ Exists fun f => And (∀ (x : (n : Nat) → E n), Membership.mem s x → Eq (f x)  …
  -/
  exact ⟨f, fs, frange, hf.continuous⟩
  /-
    🎉 no goals
  -/


theorem exists_retraction_subtype_of_isClosed {s : Set (∀ n, E n)} (hs : IsClosed s)
    (hne : s.Nonempty) :
    ∃ f : (∀ n, E n) → s, (∀ x : s, f x = x) ∧ Surjective f ∧ Continuous f := by
  obtain ⟨f, fs, rfl, f_cont⟩ :
    ∃ f : (∀ n, E n) → ∀ n, E n, (∀ x ∈ s, f x = x) ∧ range f = s ∧ Continuous f :=
    exists_retraction_of_isClosed hs hne
  /-
    case intro.intro.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    f : ((n : Nat) → E n) → (n : Nat) → E n
    f_cont : Continuous f
    hs : IsClosed (Set.range f)
    hne : (Set.range f).Nonempty
    fs : ∀ (x : (n : Nat) → E n), Membership.mem (Set.range f) x → Eq (f x) x
    ⊢ Exists fun f_1 => And (∀ (x : ↑(Set.range f)), Eq (f_1 ↑x) x) (And (Function …
  -/
  have A : ∀ x : range f, rangeFactorization f x = x := fun x ↦ Subtype.eq <| fs x x.2
  /-
    case intro.intro.intro
    E : Nat → Type u_1
    inst✝¹ : (n : Nat) → TopologicalSpace (E n)
    inst✝ : ∀ (n : Nat), DiscreteTopology (E n)
    f : ((n : Nat) → E n) → (n : Nat) → E n
    f_cont : Continuous f
    hs : IsClosed (Set.range f)
    hne : (Set.range f).Nonempty
    fs : ∀ (x : (n : Nat) → E n), Membership.mem (Set.range f) x → Eq (f x) x
    A : ∀ (x : ↑(Set.range f)), Eq (Set.rangeFactorization f ↑x) x
    ⊢ Exists fun f_1 => And (∀ (x : ↑(Set.range f)), Eq (f_1 ↑x) x) (And (Function …
  -/
  exact ⟨rangeFactorization f, A, fun x => ⟨x, A x⟩, f_cont.subtype_mk _⟩
  /-
    🎉 no goals
  -/


/-- Any nonempty complete second countable metric space is the continuous image of the
fundamental space `ℕ → ℕ`. For a version of this theorem in the context of Polish spaces, see
`exists_nat_nat_continuous_surjective_of_polishSpace`. -/
theorem exists_nat_nat_continuous_surjective_of_completeSpace (α : Type*) [MetricSpace α]
    [CompleteSpace α] [SecondCountableTopology α] [Nonempty α] :
    ∃ f : (ℕ → ℕ) → α, Continuous f ∧ Surjective f := by
  /- First, we define a surjective map from a closed subset `s` of `ℕ → ℕ`. Then, we compose
    this map with a retraction of `ℕ → ℕ` onto `s` to obtain the desired map.
    Let us consider a dense sequence `u` in `α`. Then `s` is the set of sequences `xₙ` such that the
    balls `closedBall (u xₙ) (1/2^n)` have a nonempty intersection. This set is closed,
    and we define `f x` there to be the unique point in the intersection.
    This function is continuous and surjective by design. -/
  /-
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  letI : MetricSpace (ℕ → ℕ) := PiNat.metricSpaceNatNat
  /-
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  have I0 : (0 : ℝ) < 1 / 2 := by norm_num
  /-
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    I0 : LT.lt 0 (1 / 2)
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  have I1 : (1 / 2 : ℝ) < 1 := by norm_num
  /-
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    I0 : LT.lt 0 (1 / 2)
    I1 : LT.lt (1 / 2) 1
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  rcases exists_dense_seq α with ⟨u, hu⟩
  /-
    case intro
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    I0 : LT.lt 0 (1 / 2)
    I1 : LT.lt (1 / 2) 1
    u : Nat → α
    hu : DenseRange u
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  let s : Set (ℕ → ℕ) := { x | (⋂ n : ℕ, closedBall (u (x n)) ((1 / 2) ^ n)).Nonempty }
  /-
    case intro
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    I0 : LT.lt 0 (1 / 2)
    I1 : LT.lt (1 / 2) 1
    u : Nat → α
    hu : DenseRange u
    s : Set (Nat → Nat) := setOf fun x => (Set.iInter fun n => Metric.closedBall ( …
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  let g : s → α := fun x => x.2.some
  have A : ∀ (x : s) (n : ℕ), dist (g x) (u ((x : ℕ → ℕ) n)) ≤ (1 / 2) ^ n := fun x n =>
    (mem_iInter.1 x.2.some_mem n : _)
  have g_cont : Continuous g := by
    refine continuous_iff_continuousAt.2 fun y => ?_
    refine continuousAt_of_locally_lipschitz zero_lt_one 4 fun x hxy => ?_
    rcases eq_or_ne x y with (rfl | hne)
    · simp
    have hne' : x.1 ≠ y.1 := Subtype.coe_injective.ne hne
    have dist' : dist x y = dist x.1 y.1 := rfl
    let n := firstDiff x.1 y.1 - 1
    have diff_pos : 0 < firstDiff x.1 y.1 := by
      by_contra! h
      apply apply_firstDiff_ne hne'
      rw [Nat.le_zero.1 h]
      apply apply_eq_of_dist_lt _ le_rfl
      rw [pow_zero]
      exact hxy
    have hn : firstDiff x.1 y.1 = n + 1 := (Nat.succ_pred_eq_of_pos diff_pos).symm
    rw [dist', dist_eq_of_ne hne', hn]
    have B : x.1 n = y.1 n := mem_cylinder_firstDiff x.1 y.1 n (Nat.pred_lt diff_pos.ne')
    calc
      dist (g x) (g y) ≤ dist (g x) (u (x.1 n)) + dist (g y) (u (x.1 n)) :=
        dist_triangle_right _ _ _
      _ = dist (g x) (u (x.1 n)) + dist (g y) (u (y.1 n)) := by rw [← B]
      _ ≤ (1 / 2) ^ n + (1 / 2) ^ n := add_le_add (A x n) (A y n)
      _ = 4 * (1 / 2) ^ (n + 1) := by ring
  have g_surj : Surjective g := fun y ↦ by
    have : ∀ n : ℕ, ∃ j, y ∈ closedBall (u j) ((1 / 2) ^ n) := fun n ↦ by
      rcases hu.exists_dist_lt y (by simp : (0 : ℝ) < (1 / 2) ^ n) with ⟨j, hj⟩
      exact ⟨j, hj.le⟩
    choose x hx using this
    have I : (⋂ n : ℕ, closedBall (u (x n)) ((1 / 2) ^ n)).Nonempty := ⟨y, mem_iInter.2 hx⟩
    refine ⟨⟨x, I⟩, ?_⟩
    refine dist_le_zero.1 ?_
    have J : ∀ n : ℕ, dist (g ⟨x, I⟩) y ≤ (1 / 2) ^ n + (1 / 2) ^ n := fun n =>
      calc
        dist (g ⟨x, I⟩) y ≤ dist (g ⟨x, I⟩) (u (x n)) + dist y (u (x n)) :=
          dist_triangle_right _ _ _
        _ ≤ (1 / 2) ^ n + (1 / 2) ^ n := add_le_add (A ⟨x, I⟩ n) (hx n)
    have L : Tendsto (fun n : ℕ => (1 / 2 : ℝ) ^ n + (1 / 2) ^ n) atTop (𝓝 (0 + 0)) :=
      (tendsto_pow_atTop_nhds_zero_of_lt_one I0.le I1).add
        (tendsto_pow_atTop_nhds_zero_of_lt_one I0.le I1)
    rw [add_zero] at L
    exact ge_of_tendsto' L J
  have s_closed : IsClosed s := by
    refine isClosed_iff_clusterPt.mpr fun x hx ↦ ?_
    have L : Tendsto (fun n : ℕ => diam (closedBall (u (x n)) ((1 / 2) ^ n))) atTop (𝓝 0) := by
      have : Tendsto (fun n : ℕ => (2 : ℝ) * (1 / 2) ^ n) atTop (𝓝 (2 * 0)) :=
        (tendsto_pow_atTop_nhds_zero_of_lt_one I0.le I1).const_mul _
      rw [mul_zero] at this
      exact
        squeeze_zero (fun n => diam_nonneg) (fun n => diam_closedBall (pow_nonneg I0.le _)) this
    refine nonempty_iInter_of_nonempty_biInter (fun n => isClosed_ball)
      (fun n => isBounded_closedBall) (fun N ↦ ?_) L
    obtain ⟨y, hxy, ys⟩ : ∃ y, y ∈ ball x ((1 / 2) ^ N) ∩ s :=
      clusterPt_principal_iff.1 hx _ (ball_mem_nhds x (pow_pos I0 N))
    have E :
      ⋂ (n : ℕ) (H : n ≤ N), closedBall (u (x n)) ((1 / 2) ^ n) =
        ⋂ (n : ℕ) (H : n ≤ N), closedBall (u (y n)) ((1 / 2) ^ n) := by
      refine iInter_congr fun n ↦ iInter_congr fun hn ↦ ?_
      have : x n = y n := apply_eq_of_dist_lt (mem_ball'.1 hxy) hn
      rw [this]
    rw [E]
    apply Nonempty.mono _ ys
    apply iInter_subset_iInter₂
  obtain ⟨f, -, f_surj, f_cont⟩ :
    ∃ f : (ℕ → ℕ) → s, (∀ x : s, f x = x) ∧ Surjective f ∧ Continuous f := by
    apply exists_retraction_subtype_of_isClosed s_closed
    simpa only [nonempty_coe_sort] using g_surj.nonempty
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝³ : MetricSpace α
    inst✝² : CompleteSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : Nonempty α
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    I0 : LT.lt 0 (1 / 2)
    I1 : LT.lt (1 / 2) 1
    u : Nat → α
    hu : DenseRange u
    s : Set (Nat → Nat) := setOf fun x => (Set.iInter fun n => Metric.closedBall ( …
    g : ↑s → α := fun x => Set.Nonempty.some ⋯
    A : ∀ (x : ↑s) (n : Nat), LE.le (Dist.dist (g x) (u (↑x n))) (HPow.hPow (1 / 2 …
    g_cont : Continuous g
    g_surj : Function.Surjective g
    s_closed : IsClosed s
    f : (Nat → Nat) → ↑s
    f_surj : Function.Surjective f
    f_cont : Continuous f
    ⊢ Exists fun f => And (Continuous f) (Function.Surjective f)
  -/
  exact ⟨g ∘ f, g_cont.comp f_cont, g_surj.comp f_surj⟩
  /-
    🎉 no goals
  -/


/-- Given a countable family of metric spaces, one may put a distance on their product `Π i, E i`.
It is highly non-canonical, though, and therefore not registered as a global instance.
The distance we use here is `dist x y = ∑' i, min (1/2)^(encode i) (dist (x i) (y i))`. -/
protected def dist : Dist (∀ i, F i) :=
  ⟨fun x y => ∑' i : ι, min ((1 / 2) ^ encode i) (dist (x i) (y i))⟩


theorem dist_eq_tsum (x y : ∀ i, F i) :
    dist x y = ∑' i : ι, min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) :=
  rfl


theorem dist_summable (x y : ∀ i, F i) :
    Summable fun i : ι => min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) := by
  refine .of_nonneg_of_le (fun i => ?_) (fun i => min_le_left _ _)
    summable_geometric_two_encode
  /-
    ι : Type u_2
    inst✝¹ : Encodable ι
    F : ι → Type u_3
    inst✝ : (i : ι) → MetricSpace (F i)
    x y : (i : ι) → F i
    i : ι
    ⊢ LE.le 0 (Min.min (HPow.hPow (1 / 2) (Encodable.encode i)) (Dist.dist (x i) ( …
  -/
  exact le_min (pow_nonneg (by norm_num) _) dist_nonneg
  /-
    🎉 no goals
  -/


theorem min_dist_le_dist_pi (x y : ∀ i, F i) (i : ι) :
    min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) ≤ dist x y :=
                                                      /-
                                                        ι : Type u_2
                                                        inst✝¹ : Encodable ι
                                                        F : ι → Type u_3
                                                        inst✝ : (i : ι) → MetricSpace (F i)
                                                        x y : (i : ι) → F i
                                                        i j : ι
                                                        x✝ : Ne j i
                                                        ⊢ LE.le 0 (HPow.hPow (1 / 2) (Encodable.encode j))
                                                      -/
  le_tsum (dist_summable x y) i fun j _ => le_min (by simp) dist_nonneg
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem dist_le_dist_pi_of_dist_lt {x y : ∀ i, F i} {i : ι} (h : dist x y < (1 / 2) ^ encode i) :
    dist (x i) (y i) ≤ dist x y := by
  /-
    ι : Type u_2
    inst✝¹ : Encodable ι
    F : ι → Type u_3
    inst✝ : (i : ι) → MetricSpace (F i)
    x y : (i : ι) → F i
    i : ι
    h : LT.lt (Dist.dist x y) (HPow.hPow (1 / 2) (Encodable.encode i))
    ⊢ LE.le (Dist.dist (x i) (y i)) (Dist.dist x y)
  -/
  simpa only [not_le.2 h, false_or] using min_le_iff.1 (min_dist_le_dist_pi x y i)
  /-
    🎉 no goals
  -/


/-- Given a countable family of metric spaces, one may put a distance on their product `Π i, E i`,
defining the right topology and uniform structure. It is highly non-canonical, though, and therefore
not registered as a global instance.
The distance we use here is `dist x y = ∑' n, min (1/2)^(encode i) (dist (x n) (y n))`. -/
protected def metricSpace : MetricSpace (∀ i, F i) where
                    /-
                      E : Nat → Type u_1
                      ι : Type u_2
                      inst✝¹ : Encodable ι
                      F : ι → Type u_3
                      inst✝ : (i : ι) → MetricSpace (F i)
                      x : (i : ι) → F i
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp [dist_eq_tsum]
                    /-
                      🎉 no goals
                    -/
                      /-
                        E : Nat → Type u_1
                        ι : Type u_2
                        inst✝¹ : Encodable ι
                        F : ι → Type u_3
                        inst✝ : (i : ι) → MetricSpace (F i)
                        x y : (i : ι) → F i
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  dist_comm x y := by simp [dist_eq_tsum, dist_comm]
                      /-
                        🎉 no goals
                      -/
  dist_triangle x y z :=
    have I : ∀ i, min ((1 / 2) ^ encode i : ℝ) (dist (x i) (z i)) ≤
        min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) +
          min ((1 / 2) ^ encode i : ℝ) (dist (y i) (z i)) := fun i =>
      calc
        min ((1 / 2) ^ encode i : ℝ) (dist (x i) (z i)) ≤
            min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i) + dist (y i) (z i)) :=
          min_le_min le_rfl (dist_triangle _ _ _)
        _ = min ((1 / 2) ^ encode i : ℝ) (min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) +
              min ((1 / 2) ^ encode i : ℝ) (dist (y i) (z i))) := by
          convert congr_arg ((↑) : ℝ≥0 → ℝ)
            (min_add_distrib ((1 / 2 : ℝ≥0) ^ encode i) (nndist (x i) (y i))
              (nndist (y i) (z i)))
        _ ≤ min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) +
              min ((1 / 2) ^ encode i : ℝ) (dist (y i) (z i)) :=
          min_le_right _ _
    calc dist x z ≤ ∑' i, (min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) +
          min ((1 / 2) ^ encode i : ℝ) (dist (y i) (z i))) :=
        tsum_le_tsum I (dist_summable x z) ((dist_summable x y).add (dist_summable y z))
      _ = dist x y + dist y z := tsum_add (dist_summable x y) (dist_summable y z)
  eq_of_dist_eq_zero hxy := by
    /-
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      x✝ y✝ : (i : ι) → F i
      hxy : Eq (Dist.dist x✝ y✝) 0
      ⊢ Eq x✝ y✝
    -/
    ext1 n
    /-
      case h
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      x✝ y✝ : (i : ι) → F i
      hxy : Eq (Dist.dist x✝ y✝) 0
      n : ι
      ⊢ Eq (x✝ n) (y✝ n)
    -/
    rw [← dist_le_zero, ← hxy]
    /-
      case h
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      x✝ y✝ : (i : ι) → F i
      hxy : Eq (Dist.dist x✝ y✝) 0
      n : ι
      ⊢ LE.le (Dist.dist (x✝ n) (y✝ n)) (Dist.dist x✝ y✝)
    -/
    apply dist_le_dist_pi_of_dist_lt
    /-
      case h.h
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      x✝ y✝ : (i : ι) → F i
      hxy : Eq (Dist.dist x✝ y✝) 0
      n : ι
      ⊢ LT.lt (Dist.dist x✝ y✝) (HPow.hPow (1 / 2) (Encodable.encode n))
    -/
    rw [hxy]
    /-
      case h.h
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      x✝ y✝ : (i : ι) → F i
      hxy : Eq (Dist.dist x✝ y✝) 0
      n : ι
      ⊢ LT.lt 0 (HPow.hPow (1 / 2) (Encodable.encode n))
    -/
    /-
      E : Nat → Type u_1
      ι : Type u_2
      inst✝¹ : Encodable ι
      F : ι → Type u_3
      inst✝ : (i : ι) → MetricSpace (F i)
      ⊢ Eq (iInf fun i => iInf fun i_1 => iInf fun x => Filter.principal (setOf fun  …
    -/
    simp
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        ⊢ LE.le (iInf fun i => iInf fun i_1 => iInf fun x => Filter.principal (setOf f …
      -/
    /-
      🎉 no goals
    -/
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        ⊢ ∀ (i : Real), LT.lt 0 i → Membership.mem (iInf fun i => iInf fun i_2 => iInf …
      -/
  toUniformSpace := Pi.uniformSpace _
  uniformity_dist := by
    simp only [Pi.uniformity, comap_iInf, gt_iff_lt, preimage_setOf_eq, comap_principal,
      PseudoMetricSpace.uniformity_dist]
    apply le_antisymm
    · simp only [le_iInf_iff, le_principal_iff]
      intro ε εpos
      classical
      obtain ⟨K, hK⟩ :
        ∃ K : Finset ι, (∑' i : { j // j ∉ K }, (1 / 2 : ℝ) ^ encode (i : ι)) < ε / 2 :=
        ((tendsto_order.1 (tendsto_tsum_compl_atTop_zero fun i : ι => (1 / 2 : ℝ) ^ encode i)).2 _
            (half_pos εpos)).exists
      obtain ⟨δ, δpos, hδ⟩ : ∃ δ : ℝ, 0 < δ ∧ (K.card : ℝ) * δ < ε / 2 :=
        exists_pos_mul_lt (half_pos εpos) _
      apply @mem_iInf_of_iInter _ _ _ _ _ K.finite_toSet fun i =>
          { p : (∀ i : ι, F i) × ∀ i : ι, F i | dist (p.fst i) (p.snd i) < δ }
      · rintro ⟨i, hi⟩
        refine mem_iInf_of_mem δ (mem_iInf_of_mem δpos ?_)
        simp only [Prod.forall, imp_self, mem_principal, Subset.rfl]
      · rintro ⟨x, y⟩ hxy
        simp only [mem_iInter, mem_setOf_eq, SetCoe.forall, Finset.mem_range, Finset.mem_coe] at hxy
        calc
          dist x y = ∑' i : ι, min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i)) := rfl
          _ = (∑ i ∈ K, min ((1 / 2) ^ encode i : ℝ) (dist (x i) (y i))) +
                ∑' i : ↑(K : Set ι)ᶜ, min ((1 / 2) ^ encode (i : ι) : ℝ) (dist (x i) (y i)) :=
            (sum_add_tsum_compl (dist_summable _ _)).symm
          _ ≤ (∑ i ∈ K, dist (x i) (y i)) +
                ∑' i : ↑(K : Set ι)ᶜ, ((1 / 2) ^ encode (i : ι) : ℝ) := by
            refine add_le_add (Finset.sum_le_sum fun i _ => min_le_right _ _) ?_
            refine tsum_le_tsum (fun i => min_le_left _ _) ?_ ?_
            · apply Summable.subtype (dist_summable x y) (↑K : Set ι)ᶜ
            · apply Summable.subtype summable_geometric_two_encode (↑K : Set ι)ᶜ
          _ < (∑ _i ∈ K, δ) + ε / 2 := by
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        ⊢ LE.le (iInf fun ε => iInf fun x => Filter.principal (setOf fun p => LT.lt (D …
      -/
            apply add_lt_add_of_le_of_lt _ hK
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        ⊢ ∀ (i : ι) (i_1 : Real), LT.lt 0 i_1 → Membership.mem (iInf fun ε => iInf fun …
      -/
            refine Finset.sum_le_sum fun i hi => (hxy i ?_).le
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        i : ι
        ε : Real
        εpos : LT.lt 0 ε
        ⊢ Membership.mem (iInf fun ε => iInf fun x => Filter.principal (setOf fun p => …
      -/
            simpa using hi
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        i : ι
        ε : Real
        εpos : LT.lt 0 ε
        ⊢ Membership.mem (iInf fun x => Filter.principal (setOf fun p => LT.lt (Dist.d …
      -/
          _ ≤ ε / 2 + ε / 2 :=
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        i : ι
        ε : Real
        εpos : LT.lt 0 ε
        this : LT.lt 0 (Min.min (HPow.hPow (1 / 2) (Encodable.encode i)) ε)
        ⊢ Membership.mem (iInf fun x => Filter.principal (setOf fun p => LT.lt (Dist.d …
      -/
            (add_le_add_right (by simpa only [Finset.sum_const, nsmul_eq_mul] using hδ.le) _)
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        i : ι
        ε : Real
        εpos : LT.lt 0 ε
        this : LT.lt 0 (Min.min (HPow.hPow (1 / 2) (Encodable.encode i)) ε)
        ⊢ Membership.mem (Filter.principal (setOf fun p => LT.lt (Dist.dist p.1 p.2) ( …
      -/
          _ = ε := add_halves _
      /-
        case a
        E : Nat → Type u_1
        ι : Type u_2
        inst✝¹ : Encodable ι
        F : ι → Type u_3
        inst✝ : (i : ι) → MetricSpace (F i)
        i : ι
        ε : Real
        εpos : LT.lt 0 ε
        this : LT.lt 0 (Min.min (HPow.hPow (1 / 2) (Encodable.encode i)) ε)
        ⊢ ∀ (a b : (i : ι) → F i), LT.lt (Dist.dist a b) (HPow.hPow (1 / 2) (Encodable …
      -/
    · simp only [le_iInf_iff, le_principal_iff]
      intro i ε εpos
      refine mem_iInf_of_mem (min ((1 / 2) ^ encode i : ℝ) ε) ?_
      have : 0 < min ((1 / 2) ^ encode i : ℝ) ε := lt_min (by simp) εpos
      refine mem_iInf_of_mem this ?_
      simp only [and_imp, Prod.forall, setOf_subset_setOf, lt_min_iff, mem_principal]
      intro x y hn hε
      calc
        dist (x i) (y i) ≤ dist x y := dist_le_dist_pi_of_dist_lt hn
        _ < ε := hε



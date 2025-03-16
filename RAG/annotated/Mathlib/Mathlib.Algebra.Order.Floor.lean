/-- A `FloorSemiring` is an ordered semiring over `α` with a function
`floor : α → ℕ` satisfying `∀ (n : ℕ) (x : α), n ≤ ⌊x⌋ ↔ (n : α) ≤ x)`.
Note that many lemmas require a `LinearOrder`. Please see the above `TODO`. -/
class FloorSemiring (α) [OrderedSemiring α] where
  /-- `FloorSemiring.floor a` computes the greatest natural `n` such that `(n : α) ≤ a`. -/
  floor : α → ℕ
  /-- `FloorSemiring.ceil a` computes the least natural `n` such that `a ≤ (n : α)`. -/
  ceil : α → ℕ
  /-- `FloorSemiring.floor` of a negative element is zero. -/
  floor_of_neg {a : α} (ha : a < 0) : floor a = 0
  /-- A natural number `n` is smaller than `FloorSemiring.floor a` iff its coercion to `α` is
  smaller than `a`. -/
  gc_floor {a : α} {n : ℕ} (ha : 0 ≤ a) : n ≤ floor a ↔ (n : α) ≤ a
  /-- `FloorSemiring.ceil` is the lower adjoint of the coercion `↑ : ℕ → α`. -/
  gc_ceil : GaloisConnection ceil (↑)


instance : FloorSemiring ℕ where
  floor := id
  ceil := id
  floor_of_neg ha := (Nat.not_lt_zero _ ha).elim
  gc_floor _ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a✝ n✝ : Nat
      x✝ : LE.le 0 a✝
      ⊢ Iff (LE.le n✝ (id a✝)) (LE.le (↑n✝) a✝)
    -/
    rw [Nat.cast_id]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a✝ n✝ : Nat
      x✝ : LE.le 0 a✝
      ⊢ Iff (LE.le n✝ (id a✝)) (LE.le n✝ a✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  gc_ceil n a := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      n a : Nat
      ⊢ Iff (LE.le (id n) a) (LE.le n ↑a)
    -/
    rw [Nat.cast_id]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      n a : Nat
      ⊢ Iff (LE.le (id n) a) (LE.le n a)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- `⌊a⌋₊` is the greatest natural `n` such that `n ≤ a`. If `a` is negative, then `⌊a⌋₊ = 0`. -/
def floor : α → ℕ :=
  FloorSemiring.floor


/-- `⌈a⌉₊` is the least natural `n` such that `a ≤ n` -/
def ceil : α → ℕ :=
  FloorSemiring.ceil


@[simp]
theorem floor_nat : (Nat.floor : ℕ → ℕ) = id :=
  rfl


@[simp]
theorem ceil_nat : (Nat.ceil : ℕ → ℕ) = id :=
  rfl


@[inherit_doc]
notation "⌊" a "⌋₊" => Nat.floor a


@[inherit_doc]
notation "⌈" a "⌉₊" => Nat.ceil a


theorem le_floor_iff (ha : 0 ≤ a) : n ≤ ⌊a⌋₊ ↔ (n : α) ≤ a :=
  FloorSemiring.gc_floor ha


theorem le_floor (h : (n : α) ≤ a) : n ≤ ⌊a⌋₊ :=
  (le_floor_iff <| n.cast_nonneg.trans h).2 h


theorem gc_ceil_coe : GaloisConnection (ceil : α → ℕ) (↑) :=
  FloorSemiring.gc_ceil


@[simp]
theorem ceil_le : ⌈a⌉₊ ≤ n ↔ a ≤ n :=
  gc_ceil_coe _ _


theorem floor_lt (ha : 0 ≤ a) : ⌊a⌋₊ < n ↔ a < n :=
  lt_iff_lt_of_le_iff_le <| le_floor_iff ha


theorem floor_lt_one (ha : 0 ≤ a) : ⌊a⌋₊ < 1 ↔ a < 1 :=
                            /-
                              α : Type u_2
                              inst✝¹ : LinearOrderedSemiring α
                              inst✝ : FloorSemiring α
                              a : α
                              ha : LE.le 0 a
                              ⊢ Iff (LT.lt a ↑1) (LT.lt a 1)
                            -/
  (floor_lt ha).trans <| by rw [Nat.cast_one]
                            /-
                              🎉 no goals
                            -/


theorem lt_of_floor_lt (h : ⌊a⌋₊ < n) : a < n :=
  lt_of_not_le fun h' => (le_floor h').not_lt h


theorem lt_one_of_floor_lt_one (h : ⌊a⌋₊ < 1) : a < 1 := mod_cast lt_of_floor_lt h


theorem floor_le (ha : 0 ≤ a) : (⌊a⌋₊ : α) ≤ a :=
  (le_floor_iff ha).1 le_rfl


theorem lt_succ_floor (a : α) : a < ⌊a⌋₊.succ :=
  lt_of_floor_lt <| Nat.lt_succ_self _


@[bound]
                                                      /-
                                                        α : Type u_2
                                                        inst✝¹ : LinearOrderedSemiring α
                                                        inst✝ : FloorSemiring α
                                                        a : α
                                                        ⊢ LT.lt a (HAdd.hAdd (↑(Nat.floor a)) 1)
                                                      -/
theorem lt_floor_add_one (a : α) : a < ⌊a⌋₊ + 1 := by simpa using lt_succ_floor a
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem floor_natCast (n : ℕ) : ⌊(n : α)⌋₊ = n :=
  eq_of_forall_le_iff fun a => by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      n a : Nat
      ⊢ Iff (LE.le a (Nat.floor ↑n)) (LE.le a n)
    -/
    rw [le_floor_iff, Nat.cast_le]
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      n a : Nat
      ⊢ LE.le 0 ↑n
    -/
    exact n.cast_nonneg
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-08")] alias floor_coe := floor_natCast


@[simp]
                                          /-
                                            α : Type u_2
                                            inst✝¹ : LinearOrderedSemiring α
                                            inst✝ : FloorSemiring α
                                            ⊢ Eq (Nat.floor 0) 0
                                          -/
theorem floor_zero : ⌊(0 : α)⌋₊ = 0 := by rw [← Nat.cast_zero, floor_natCast]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝¹ : LinearOrderedSemiring α
                                           inst✝ : FloorSemiring α
                                           ⊢ Eq (Nat.floor 1) 1
                                         -/
theorem floor_one : ⌊(1 : α)⌋₊ = 1 := by rw [← Nat.cast_one, floor_natCast]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem floor_ofNat (n : ℕ) [n.AtLeastTwo] : ⌊(ofNat(n) : α)⌋₊ = ofNat(n) :=
  Nat.floor_natCast _


theorem floor_of_nonpos (ha : a ≤ 0) : ⌊a⌋₊ = 0 :=
  ha.lt_or_eq.elim FloorSemiring.floor_of_neg <| by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le a 0
      ⊢ Eq a 0 → Eq (Nat.floor a) 0
    -/
    rintro rfl
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      ha : LE.le 0 0
      ⊢ Eq (Nat.floor 0) 0
    -/
    exact floor_zero
    /-
      🎉 no goals
    -/


theorem floor_mono : Monotone (floor : α → ℕ) := fun a b h => by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    h : LE.le a b
    ⊢ LE.le (Nat.floor a) (Nat.floor b)
  -/
  obtain ha | ha := le_total a 0
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LE.le a b
      ha : LE.le a 0
      ⊢ LE.le (Nat.floor a) (Nat.floor b)
    -/
  · rw [floor_of_nonpos ha]
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LE.le a b
      ha : LE.le a 0
      ⊢ LE.le 0 (Nat.floor b)
    -/
    exact Nat.zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LE.le a b
      ha : LE.le 0 a
      ⊢ LE.le (Nat.floor a) (Nat.floor b)
    -/
  · exact le_floor ((floor_le ha).trans h)
    /-
      🎉 no goals
    -/


@[gcongr, bound] lemma floor_le_floor (hab : a ≤ b) : ⌊a⌋₊ ≤ ⌊b⌋₊ := floor_mono hab


theorem le_floor_iff' (hn : n ≠ 0) : n ≤ ⌊a⌋₊ ↔ (n : α) ≤ a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LE.le n (Nat.floor a)) (LE.le (↑n) a)
  -/
  obtain ha | ha := le_total a 0
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      hn : Ne n 0
      ha : LE.le a 0
      ⊢ Iff (LE.le n (Nat.floor a)) (LE.le (↑n) a)
    -/
  · rw [floor_of_nonpos ha]
    exact
      iff_of_false (Nat.pos_of_ne_zero hn).not_le
        (not_le_of_lt <| ha.trans_lt <| cast_pos.2 <| Nat.pos_of_ne_zero hn)
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      hn : Ne n 0
      ha : LE.le 0 a
      ⊢ Iff (LE.le n (Nat.floor a)) (LE.le (↑n) a)
    -/
  · exact le_floor_iff ha
    /-
      🎉 no goals
    -/


@[simp]
theorem one_le_floor_iff (x : α) : 1 ≤ ⌊x⌋₊ ↔ 1 ≤ x :=
  mod_cast @le_floor_iff' α _ _ x 1 one_ne_zero


theorem floor_lt' (hn : n ≠ 0) : ⌊a⌋₊ < n ↔ a < n :=
  lt_iff_lt_of_le_iff_le <| le_floor_iff' hn


theorem floor_pos : 0 < ⌊a⌋₊ ↔ 1 ≤ a := by
  -- Porting note: broken `convert le_floor_iff' Nat.one_ne_zero`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Iff (LT.lt 0 (Nat.floor a)) (LE.le 1 a)
  -/
  rw [Nat.lt_iff_add_one_le, zero_add, le_floor_iff' Nat.one_ne_zero, cast_one]
  /-
    🎉 no goals
  -/


theorem pos_of_floor_pos (h : 0 < ⌊a⌋₊) : 0 < a :=
                                                          /-
                                                            α : Type u_2
                                                            inst✝¹ : LinearOrderedSemiring α
                                                            inst✝ : FloorSemiring α
                                                            a : α
                                                            h : LT.lt 0 (Nat.floor a)
                                                            ha : LE.le a 0
                                                            ⊢ LT.lt 0 0
                                                          -/
  (le_or_lt a 0).resolve_left fun ha => lt_irrefl 0 <| by rwa [floor_of_nonpos ha] at h
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem lt_of_lt_floor (h : n < ⌊a⌋₊) : ↑n < a :=
  (Nat.cast_lt.2 h).trans_le <| floor_le (pos_of_floor_pos <| (Nat.zero_le n).trans_lt h).le


theorem floor_le_of_le (h : a ≤ n) : ⌊a⌋₊ ≤ n :=
  le_imp_le_iff_lt_imp_lt.2 lt_of_lt_floor h


theorem floor_le_one_of_le_one (h : a ≤ 1) : ⌊a⌋₊ ≤ 1 :=
  floor_le_of_le <| h.trans_eq <| Nat.cast_one.symm


@[simp]
theorem floor_eq_zero : ⌊a⌋₊ = 0 ↔ a < 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Iff (Eq (Nat.floor a) 0) (LT.lt a 1)
  -/
  rw [← lt_one_iff, ← @cast_one α]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Iff (LT.lt (Nat.floor a) 1) (LT.lt a ↑1)
  -/
  exact floor_lt' Nat.one_ne_zero
  /-
    🎉 no goals
  -/


theorem floor_eq_iff (ha : 0 ≤ a) : ⌊a⌋₊ = n ↔ ↑n ≤ a ∧ a < ↑n + 1 := by
  rw [← le_floor_iff ha, ← Nat.cast_one, ← Nat.cast_add, ← floor_lt ha, Nat.lt_add_one_iff,
    le_antisymm_iff, and_comm]


theorem floor_eq_iff' (hn : n ≠ 0) : ⌊a⌋₊ = n ↔ ↑n ≤ a ∧ a < ↑n + 1 := by
  rw [← le_floor_iff' hn, ← Nat.cast_one, ← Nat.cast_add, ← floor_lt' (Nat.add_one_ne_zero n),
    Nat.lt_add_one_iff, le_antisymm_iff, and_comm]


theorem floor_eq_on_Ico (n : ℕ) : ∀ a ∈ (Set.Ico n (n + 1) : Set α), ⌊a⌋₊ = n := fun _ ⟨h₀, h₁⟩ =>
  (floor_eq_iff <| n.cast_nonneg.trans h₀).mpr ⟨h₀, h₁⟩


theorem floor_eq_on_Ico' (n : ℕ) :
    ∀ a ∈ (Set.Ico n (n + 1) : Set α), (⌊a⌋₊ : α) = n :=
  fun x hx => mod_cast floor_eq_on_Ico n x hx


@[simp]
theorem preimage_floor_zero : (floor : α → ℕ) ⁻¹' {0} = Iio 1 :=
  ext fun _ => floor_eq_zero

-- Porting note: in mathlib3 there was no need for the type annotation in `(n:α)`

theorem preimage_floor_of_ne_zero {n : ℕ} (hn : n ≠ 0) :
    (floor : α → ℕ) ⁻¹' {n} = Ico (n : α) (n + 1) :=
  ext fun _ => floor_eq_iff' hn


theorem lt_ceil : n < ⌈a⌉₊ ↔ (n : α) < a :=
  lt_iff_lt_of_le_iff_le ceil_le


theorem add_one_le_ceil_iff : n + 1 ≤ ⌈a⌉₊ ↔ (n : α) < a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ⊢ Iff (LE.le (HAdd.hAdd n 1) (Nat.ceil a)) (LT.lt (↑n) a)
  -/
  rw [← Nat.lt_ceil, Nat.add_one_le_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_le_ceil_iff : 1 ≤ ⌈a⌉₊ ↔ 0 < a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Iff (LE.le 1 (Nat.ceil a)) (LT.lt 0 a)
  -/
  rw [← zero_add 1, Nat.add_one_le_ceil_iff, Nat.cast_zero]
  /-
    🎉 no goals
  -/


@[bound]
theorem ceil_le_floor_add_one (a : α) : ⌈a⌉₊ ≤ ⌊a⌋₊ + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ LE.le (Nat.ceil a) (HAdd.hAdd (Nat.floor a) 1)
  -/
  rw [ceil_le, Nat.cast_add, Nat.cast_one]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ LE.le a (HAdd.hAdd (↑(Nat.floor a)) 1)
  -/
  exact (lt_floor_add_one a).le
  /-
    🎉 no goals
  -/


@[bound]
theorem le_ceil (a : α) : a ≤ ⌈a⌉₊ :=
  ceil_le.1 le_rfl


@[simp]
theorem ceil_intCast {α : Type*} [LinearOrderedRing α] [FloorSemiring α] (z : ℤ) :
    ⌈(z : α)⌉₊ = z.toNat :=
  eq_of_forall_ge_iff fun a => by
    /-
      α : Type u_4
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorSemiring α
      z : Int
      a : Nat
      ⊢ Iff (LE.le (Nat.ceil ↑z) a) (LE.le z.toNat a)
    -/
    simp only [ceil_le, Int.toNat_le]
    /-
      α : Type u_4
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorSemiring α
      z : Int
      a : Nat
      ⊢ Iff (LE.le ↑z ↑a) (LE.le z ↑a)
    -/
    norm_cast
    /-
      🎉 no goals
    -/


@[simp]
theorem ceil_natCast (n : ℕ) : ⌈(n : α)⌉₊ = n :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedSemiring α
                                    inst✝ : FloorSemiring α
                                    n a : Nat
                                    ⊢ Iff (LE.le (Nat.ceil ↑n) a) (LE.le n a)
                                  -/
  eq_of_forall_ge_iff fun a => by rw [ceil_le, cast_le]
                                  /-
                                    🎉 no goals
                                  -/


theorem ceil_mono : Monotone (ceil : α → ℕ) :=
  gc_ceil_coe.monotone_l


@[gcongr, bound] lemma ceil_le_ceil (hab : a ≤ b) : ⌈a⌉₊ ≤ ⌈b⌉₊ := ceil_mono hab


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝¹ : LinearOrderedSemiring α
                                           inst✝ : FloorSemiring α
                                           ⊢ Eq (Nat.ceil 0) 0
                                         -/
theorem ceil_zero : ⌈(0 : α)⌉₊ = 0 := by rw [← Nat.cast_zero, ceil_natCast]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                        /-
                                          α : Type u_2
                                          inst✝¹ : LinearOrderedSemiring α
                                          inst✝ : FloorSemiring α
                                          ⊢ Eq (Nat.ceil 1) 1
                                        -/
theorem ceil_one : ⌈(1 : α)⌉₊ = 1 := by rw [← Nat.cast_one, ceil_natCast]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem ceil_ofNat (n : ℕ) [n.AtLeastTwo] : ⌈(ofNat(n) : α)⌉₊ = ofNat(n) := ceil_natCast n


@[simp]
                                              /-
                                                α : Type u_2
                                                inst✝¹ : LinearOrderedSemiring α
                                                inst✝ : FloorSemiring α
                                                a : α
                                                ⊢ Iff (Eq (Nat.ceil a) 0) (LE.le a 0)
                                              -/
theorem ceil_eq_zero : ⌈a⌉₊ = 0 ↔ a ≤ 0 := by rw [← Nat.le_zero, ceil_le, Nat.cast_zero]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                          /-
                                            α : Type u_2
                                            inst✝¹ : LinearOrderedSemiring α
                                            inst✝ : FloorSemiring α
                                            a : α
                                            ⊢ Iff (LT.lt 0 (Nat.ceil a)) (LT.lt 0 a)
                                          -/
theorem ceil_pos : 0 < ⌈a⌉₊ ↔ 0 < a := by rw [lt_ceil, cast_zero]
                                          /-
                                            🎉 no goals
                                          -/


theorem lt_of_ceil_lt (h : ⌈a⌉₊ < n) : a < n :=
  (le_ceil a).trans_lt (Nat.cast_lt.2 h)


theorem le_of_ceil_le (h : ⌈a⌉₊ ≤ n) : a ≤ n :=
  (le_ceil a).trans (Nat.cast_le.2 h)


@[bound]
theorem floor_le_ceil (a : α) : ⌊a⌋₊ ≤ ⌈a⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ LE.le (Nat.floor a) (Nat.ceil a)
  -/
  obtain ha | ha := le_total a 0
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le a 0
      ⊢ LE.le (Nat.floor a) (Nat.ceil a)
    -/
  · rw [floor_of_nonpos ha]
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le a 0
      ⊢ LE.le 0 (Nat.ceil a)
    -/
    exact Nat.zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      ⊢ LE.le (Nat.floor a) (Nat.ceil a)
    -/
  · exact cast_le.1 ((floor_le ha).trans <| le_ceil _)
    /-
      🎉 no goals
    -/


theorem floor_lt_ceil_of_lt_of_pos {a b : α} (h : a < b) (h' : 0 < b) : ⌊a⌋₊ < ⌈b⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    h : LT.lt a b
    h' : LT.lt 0 b
    ⊢ LT.lt (Nat.floor a) (Nat.ceil b)
  -/
  rcases le_or_lt 0 a with (ha | ha)
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LT.lt a b
      h' : LT.lt 0 b
      ha : LE.le 0 a
      ⊢ LT.lt (Nat.floor a) (Nat.ceil b)
    -/
  · rw [floor_lt ha]
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LT.lt a b
      h' : LT.lt 0 b
      ha : LE.le 0 a
      ⊢ LT.lt a ↑(Nat.ceil b)
    -/
    exact h.trans_le (le_ceil _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a b : α
      h : LT.lt a b
      h' : LT.lt 0 b
      ha : LT.lt a 0
      ⊢ LT.lt (Nat.floor a) (Nat.ceil b)
    -/
  · rwa [floor_of_nonpos ha.le, lt_ceil, Nat.cast_zero]
    /-
      🎉 no goals
    -/


theorem ceil_eq_iff (hn : n ≠ 0) : ⌈a⌉₊ = n ↔ ↑(n - 1) < a ∧ a ≤ n := by
  rw [← ceil_le, ← not_le, ← ceil_le, not_le,
    tsub_lt_iff_right (Nat.add_one_le_iff.2 (pos_iff_ne_zero.2 hn)), Nat.lt_add_one_iff,
    le_antisymm_iff, and_comm]


@[simp]
theorem preimage_ceil_zero : (Nat.ceil : α → ℕ) ⁻¹' {0} = Iic 0 :=
  ext fun _ => ceil_eq_zero

-- Porting note: in mathlib3 there was no need for the type annotation in `(↑(n - 1))`

theorem preimage_ceil_of_ne_zero (hn : n ≠ 0) : (Nat.ceil : α → ℕ) ⁻¹' {n} = Ioc (↑(n - 1) : α) n :=
  ext fun _ => ceil_eq_iff hn


@[simp]
theorem preimage_Ioo {a b : α} (ha : 0 ≤ a) :
    (Nat.cast : ℕ → α) ⁻¹' Set.Ioo a b = Set.Ioo ⌊a⌋₊ ⌈b⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ha : LE.le 0 a
    ⊢ Eq (Set.preimage Nat.cast (Set.Ioo a b)) (Set.Ioo (Nat.floor a) (Nat.ceil b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ha : LE.le 0 a
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Ioo a b)) x✝) (Membership.me …
  -/
  simp [floor_lt, lt_ceil, ha]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Ico {a b : α} : (Nat.cast : ℕ → α) ⁻¹' Set.Ico a b = Set.Ico ⌈a⌉₊ ⌈b⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ⊢ Eq (Set.preimage Nat.cast (Set.Ico a b)) (Set.Ico (Nat.ceil a) (Nat.ceil b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Ico a b)) x✝) (Membership.me …
  -/
  simp [ceil_le, lt_ceil]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Ioc {a b : α} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (Nat.cast : ℕ → α) ⁻¹' Set.Ioc a b = Set.Ioc ⌊a⌋₊ ⌊b⌋₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ Eq (Set.preimage Nat.cast (Set.Ioc a b)) (Set.Ioc (Nat.floor a) (Nat.floor b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ha : LE.le 0 a
    hb : LE.le 0 b
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Ioc a b)) x✝) (Membership.me …
  -/
  simp [floor_lt, le_floor_iff, hb, ha]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Icc {a b : α} (hb : 0 ≤ b) :
    (Nat.cast : ℕ → α) ⁻¹' Set.Icc a b = Set.Icc ⌈a⌉₊ ⌊b⌋₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    hb : LE.le 0 b
    ⊢ Eq (Set.preimage Nat.cast (Set.Icc a b)) (Set.Icc (Nat.ceil a) (Nat.floor b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    hb : LE.le 0 b
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Icc a b)) x✝) (Membership.me …
  -/
  simp [ceil_le, hb, le_floor_iff]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Ioi {a : α} (ha : 0 ≤ a) : (Nat.cast : ℕ → α) ⁻¹' Set.Ioi a = Set.Ioi ⌊a⌋₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (Set.preimage Nat.cast (Set.Ioi a)) (Set.Ioi (Nat.floor a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Ioi a)) x✝) (Membership.mem  …
  -/
  simp [floor_lt, ha]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Ici {a : α} : (Nat.cast : ℕ → α) ⁻¹' Set.Ici a = Set.Ici ⌈a⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Eq (Set.preimage Nat.cast (Set.Ici a)) (Set.Ici (Nat.ceil a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Ici a)) x✝) (Membership.mem  …
  -/
  simp [ceil_le]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Iio {a : α} : (Nat.cast : ℕ → α) ⁻¹' Set.Iio a = Set.Iio ⌈a⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ⊢ Eq (Set.preimage Nat.cast (Set.Iio a)) (Set.Iio (Nat.ceil a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Iio a)) x✝) (Membership.mem  …
  -/
  simp [lt_ceil]
  /-
    🎉 no goals
  -/

-- Porting note: changed `(coe : ℕ → α)` to `(Nat.cast : ℕ → α)`

@[simp]
theorem preimage_Iic {a : α} (ha : 0 ≤ a) : (Nat.cast : ℕ → α) ⁻¹' Set.Iic a = Set.Iic ⌊a⌋₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (Set.preimage Nat.cast (Set.Iic a)) (Set.Iic (Nat.floor a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    x✝ : Nat
    ⊢ Iff (Membership.mem (Set.preimage Nat.cast (Set.Iic a)) x✝) (Membership.mem  …
  -/
  simp [le_floor_iff, ha]
  /-
    🎉 no goals
  -/


theorem floor_add_nat (ha : 0 ≤ a) (n : ℕ) : ⌊a + n⌋₊ = ⌊a⌋₊ + n :=
  eq_of_forall_le_iff fun b => by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      n b : Nat
      ⊢ Iff (LE.le b (Nat.floor (HAdd.hAdd a ↑n))) (LE.le b (HAdd.hAdd (Nat.floor a) …
    -/
    rw [le_floor_iff (add_nonneg ha n.cast_nonneg)]
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      n b : Nat
      ⊢ Iff (LE.le (↑b) (HAdd.hAdd a ↑n)) (LE.le b (HAdd.hAdd (Nat.floor a) n))
    -/
    obtain hb | hb := le_total n b
      /-
        case inl
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        n b : Nat
        hb : LE.le n b
        ⊢ Iff (LE.le (↑b) (HAdd.hAdd a ↑n)) (LE.le b (HAdd.hAdd (Nat.floor a) n))
      -/
    · obtain ⟨d, rfl⟩ := exists_add_of_le hb
      rw [Nat.cast_add, add_comm n, add_comm (n : α), add_le_add_iff_right, add_le_add_iff_right,
        le_floor_iff ha]
      /-
        case inr
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        n b : Nat
        hb : LE.le b n
        ⊢ Iff (LE.le (↑b) (HAdd.hAdd a ↑n)) (LE.le b (HAdd.hAdd (Nat.floor a) n))
      -/
    · obtain ⟨d, rfl⟩ := exists_add_of_le hb
      /-
        case inr.intro
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        b d : Nat
        hb : LE.le b (HAdd.hAdd b d)
        ⊢ Iff (LE.le (↑b) (HAdd.hAdd a ↑(HAdd.hAdd b d))) (LE.le b (HAdd.hAdd (Nat.flo …
      -/
      rw [Nat.cast_add, add_left_comm _ b, add_left_comm _ (b : α)]
      /-
        case inr.intro
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        b d : Nat
        hb : LE.le b (HAdd.hAdd b d)
        ⊢ Iff (LE.le (↑b) (HAdd.hAdd (↑b) (HAdd.hAdd a ↑d))) (LE.le b (HAdd.hAdd b (HA …
      -/
      refine iff_of_true ?_ le_self_add
      /-
        case inr.intro
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        b d : Nat
        hb : LE.le b (HAdd.hAdd b d)
        ⊢ LE.le (↑b) (HAdd.hAdd (↑b) (HAdd.hAdd a ↑d))
      -/
      exact le_add_of_nonneg_right <| ha.trans <| le_add_of_nonneg_right d.cast_nonneg
      /-
        🎉 no goals
      -/


theorem floor_add_one (ha : 0 ≤ a) : ⌊a + 1⌋₊ = ⌊a⌋₊ + 1 := by
  -- Porting note: broken `convert floor_add_nat ha 1`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (Nat.floor (HAdd.hAdd a 1)) (HAdd.hAdd (Nat.floor a) 1)
  -/
  rw [← cast_one, floor_add_nat ha 1]
  /-
    🎉 no goals
  -/


theorem floor_add_ofNat (ha : 0 ≤ a) (n : ℕ) [n.AtLeastTwo] :
    ⌊a + ofNat(n)⌋₊ = ⌊a⌋₊ + ofNat(n) :=
  floor_add_nat ha n


@[simp]
theorem floor_sub_nat [Sub α] [OrderedSub α] [ExistsAddOfLE α] (a : α) (n : ℕ) :
    ⌊a - n⌋₊ = ⌊a⌋₊ - n := by
  /-
    α : Type u_2
    inst✝⁴ : LinearOrderedSemiring α
    inst✝³ : FloorSemiring α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : ExistsAddOfLE α
    a : α
    n : Nat
    ⊢ Eq (Nat.floor (HSub.hSub a ↑n)) (HSub.hSub (Nat.floor a) n)
  -/
  obtain ha | ha := le_total a 0
    /-
      case inl
      α : Type u_2
      inst✝⁴ : LinearOrderedSemiring α
      inst✝³ : FloorSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : ExistsAddOfLE α
      a : α
      n : Nat
      ha : LE.le a 0
      ⊢ Eq (Nat.floor (HSub.hSub a ↑n)) (HSub.hSub (Nat.floor a) n)
    -/
  · rw [floor_of_nonpos ha, floor_of_nonpos (tsub_nonpos_of_le (ha.trans n.cast_nonneg)), zero_tsub]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝⁴ : LinearOrderedSemiring α
    inst✝³ : FloorSemiring α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : ExistsAddOfLE α
    a : α
    n : Nat
    ha : LE.le 0 a
    ⊢ Eq (Nat.floor (HSub.hSub a ↑n)) (HSub.hSub (Nat.floor a) n)
  -/
  rcases le_total a n with h | h
    /-
      case inr.inl
      α : Type u_2
      inst✝⁴ : LinearOrderedSemiring α
      inst✝³ : FloorSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : ExistsAddOfLE α
      a : α
      n : Nat
      ha : LE.le 0 a
      h : LE.le a ↑n
      ⊢ Eq (Nat.floor (HSub.hSub a ↑n)) (HSub.hSub (Nat.floor a) n)
    -/
  · rw [floor_of_nonpos (tsub_nonpos_of_le h), eq_comm, tsub_eq_zero_iff_le]
    /-
      case inr.inl
      α : Type u_2
      inst✝⁴ : LinearOrderedSemiring α
      inst✝³ : FloorSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : ExistsAddOfLE α
      a : α
      n : Nat
      ha : LE.le 0 a
      h : LE.le a ↑n
      ⊢ LE.le (Nat.floor a) n
    -/
    exact Nat.cast_le.1 ((Nat.floor_le ha).trans h)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_2
      inst✝⁴ : LinearOrderedSemiring α
      inst✝³ : FloorSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : ExistsAddOfLE α
      a : α
      n : Nat
      ha : LE.le 0 a
      h : LE.le (↑n) a
      ⊢ Eq (Nat.floor (HSub.hSub a ↑n)) (HSub.hSub (Nat.floor a) n)
    -/
  · rw [eq_tsub_iff_add_eq_of_le (le_floor h), ← floor_add_nat _, tsub_add_cancel_of_le h]
    /-
      α : Type u_2
      inst✝⁴ : LinearOrderedSemiring α
      inst✝³ : FloorSemiring α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : ExistsAddOfLE α
      a : α
      n : Nat
      ha : LE.le 0 a
      h : LE.le (↑n) a
      ⊢ LE.le 0 (HSub.hSub a ↑n)
    -/
    exact le_tsub_of_add_le_left ((add_zero _).trans_le h)
    /-
      🎉 no goals
    -/


@[simp]
theorem floor_sub_one [Sub α] [OrderedSub α] [ExistsAddOfLE α] (a : α) : ⌊a - 1⌋₊ = ⌊a⌋₊ - 1 :=
  mod_cast floor_sub_nat a 1


@[simp]
theorem floor_sub_ofNat [Sub α] [OrderedSub α] [ExistsAddOfLE α] (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌊a - ofNat(n)⌋₊ = ⌊a⌋₊ - ofNat(n) :=
  floor_sub_nat a n


theorem ceil_add_nat (ha : 0 ≤ a) (n : ℕ) : ⌈a + n⌉₊ = ⌈a⌉₊ + n :=
  eq_of_forall_ge_iff fun b => by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      n b : Nat
      ⊢ Iff (LE.le (Nat.ceil (HAdd.hAdd a ↑n)) b) (LE.le (HAdd.hAdd (Nat.ceil a) n) b)
    -/
    rw [← not_lt, ← not_lt, not_iff_not, lt_ceil]
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedSemiring α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      n b : Nat
      ⊢ Iff (LT.lt (↑b) (HAdd.hAdd a ↑n)) (LT.lt b (HAdd.hAdd (Nat.ceil a) n))
    -/
    obtain hb | hb := le_or_lt n b
      /-
        case inl
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        n b : Nat
        hb : LE.le n b
        ⊢ Iff (LT.lt (↑b) (HAdd.hAdd a ↑n)) (LT.lt b (HAdd.hAdd (Nat.ceil a) n))
      -/
    · obtain ⟨d, rfl⟩ := exists_add_of_le hb
      rw [Nat.cast_add, add_comm n, add_comm (n : α), add_lt_add_iff_right, add_lt_add_iff_right,
        lt_ceil]
      /-
        case inr
        α : Type u_2
        inst✝¹ : LinearOrderedSemiring α
        inst✝ : FloorSemiring α
        a : α
        ha : LE.le 0 a
        n b : Nat
        hb : LT.lt b n
        ⊢ Iff (LT.lt (↑b) (HAdd.hAdd a ↑n)) (LT.lt b (HAdd.hAdd (Nat.ceil a) n))
      -/
    · exact iff_of_true (lt_add_of_nonneg_of_lt ha <| cast_lt.2 hb) (Nat.lt_add_left _ hb)
      /-
        🎉 no goals
      -/


theorem ceil_add_one (ha : 0 ≤ a) : ⌈a + 1⌉₊ = ⌈a⌉₊ + 1 := by
  -- Porting note: broken `convert ceil_add_nat ha 1`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (Nat.ceil (HAdd.hAdd a 1)) (HAdd.hAdd (Nat.ceil a) 1)
  -/
  rw [cast_one.symm, ceil_add_nat ha 1]
  /-
    🎉 no goals
  -/


theorem ceil_add_ofNat (ha : 0 ≤ a) (n : ℕ) [n.AtLeastTwo] :
    ⌈a + ofNat(n)⌉₊ = ⌈a⌉₊ + ofNat(n) :=
  ceil_add_nat ha n


@[bound]
theorem ceil_lt_add_one (ha : 0 ≤ a) : (⌈a⌉₊ : α) < a + 1 :=
  lt_ceil.1 <| (Nat.lt_succ_self _).trans_le (ceil_add_one ha).ge


@[bound]
theorem ceil_add_le (a b : α) : ⌈a + b⌉₊ ≤ ⌈a⌉₊ + ⌈b⌉₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ⊢ LE.le (Nat.ceil (HAdd.hAdd a b)) (HAdd.hAdd (Nat.ceil a) (Nat.ceil b))
  -/
  rw [ceil_le, Nat.cast_add]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    a b : α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd ↑(Nat.ceil a) ↑(Nat.ceil b))
  -/
             /-
               🎉 no goals
             -/
  gcongr <;> apply le_ceil
             /-
               🎉 no goals
             -/


@[bound]
theorem sub_one_lt_floor (a : α) : a - 1 < ⌊a⌋₊ :=
  sub_lt_iff_lt_add.2 <| lt_floor_add_one a


theorem floor_div_nat (a : α) (n : ℕ) : ⌊a / n⌋₊ = ⌊a⌋₊ / n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ⊢ Eq (Nat.floor (HDiv.hDiv a ↑n)) (HDiv.hDiv (Nat.floor a) n)
  -/
  rcases le_total a 0 with ha | ha
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le a 0
      ⊢ Eq (Nat.floor (HDiv.hDiv a ↑n)) (HDiv.hDiv (Nat.floor a) n)
    -/
  · rw [floor_of_nonpos, floor_of_nonpos ha]
      /-
        case inl
        α : Type u_2
        inst✝¹ : LinearOrderedSemifield α
        inst✝ : FloorSemiring α
        a : α
        n : Nat
        ha : LE.le a 0
        ⊢ Eq 0 (HDiv.hDiv 0 n)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le a 0
      ⊢ LE.le (HDiv.hDiv a ↑n) 0
    -/
    apply div_nonpos_of_nonpos_of_nonneg ha n.cast_nonneg
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ha : LE.le 0 a
    ⊢ Eq (Nat.floor (HDiv.hDiv a ↑n)) (HDiv.hDiv (Nat.floor a) n)
  -/
  obtain rfl | hn := n.eq_zero_or_pos
    /-
      case inr.inl
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      ha : LE.le 0 a
      ⊢ Eq (Nat.floor (HDiv.hDiv a ↑0)) (HDiv.hDiv (Nat.floor a) 0)
    -/
  · rw [cast_zero, div_zero, Nat.div_zero, floor_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ha : LE.le 0 a
    hn : GT.gt n 0
    ⊢ Eq (Nat.floor (HDiv.hDiv a ↑n)) (HDiv.hDiv (Nat.floor a) n)
  -/
  refine (floor_eq_iff ?_).2 ?_
    /-
      case inr.inr.refine_1
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le 0 a
      hn : GT.gt n 0
      ⊢ LE.le 0 (HDiv.hDiv a ↑n)
    -/
  · exact div_nonneg ha n.cast_nonneg
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.refine_2
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ha : LE.le 0 a
    hn : GT.gt n 0
    ⊢ And (LE.le (↑(HDiv.hDiv (Nat.floor a) n)) (HDiv.hDiv a ↑n)) (LT.lt (HDiv.hDi …
  -/
  constructor
    /-
      case inr.inr.refine_2.left
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le 0 a
      hn : GT.gt n 0
      ⊢ LE.le (↑(HDiv.hDiv (Nat.floor a) n)) (HDiv.hDiv a ↑n)
    -/
  · exact cast_div_le.trans (div_le_div_of_nonneg_right (floor_le ha) n.cast_nonneg)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.refine_2.right
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    a : α
    n : Nat
    ha : LE.le 0 a
    hn : GT.gt n 0
    ⊢ LT.lt (HDiv.hDiv a ↑n) (HAdd.hAdd (↑(HDiv.hDiv (Nat.floor a) n)) 1)
  -/
  rw [div_lt_iff₀, add_mul, one_mul, ← cast_mul, ← cast_add, ← floor_lt ha]
    /-
      case inr.inr.refine_2.right
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le 0 a
      hn : GT.gt n 0
      ⊢ LT.lt (Nat.floor a) (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Nat.floor a) n) n) n)
    -/
  · exact lt_div_mul_add hn
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.refine_2.right
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : FloorSemiring α
      a : α
      n : Nat
      ha : LE.le 0 a
      hn : GT.gt n 0
      ⊢ LT.lt 0 ↑n
    -/
  · exact cast_pos.2 hn
    /-
      🎉 no goals
    -/


theorem floor_div_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌊a / ofNat(n)⌋₊ = ⌊a⌋₊ / ofNat(n) :=
  floor_div_nat a n


/-- Natural division is the floor of field division. -/
theorem floor_div_eq_div (m n : ℕ) : ⌊(m : α) / n⌋₊ = m / n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    m n : Nat
    ⊢ Eq (Nat.floor (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv m n)
  -/
  convert floor_div_nat (m : α) n
  /-
    case h.e'_3.h.e'_5
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : FloorSemiring α
    m n : Nat
    ⊢ Eq m (Nat.floor ↑m)
  -/
  rw [m.floor_natCast]
  /-
    🎉 no goals
  -/


lemma mul_lt_floor (hb₀ : 0 < b) (hb : b < 1) (hba : ⌈b / (1 - b)⌉₊ ≤ a) : b * a < ⌊a⌋₊ := by
  calc
    b * a < b * (⌊a⌋₊ + 1) := by gcongr; exacts [hb₀, lt_floor_add_one _]
    _ ≤ ⌊a⌋₊ := by
      rw [_root_.mul_add_one, ← le_sub_iff_add_le', ← one_sub_mul, ← div_le_iff₀' (by linarith),
        ← ceil_le]
      exact le_floor hba


lemma ceil_lt_mul (hb : 1 < b) (hba : ⌈(b - 1)⁻¹⌉₊ / b < a) : ⌈a⌉₊ < b * a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorSemiring α
    a b : α
    hb : LT.lt 1 b
    hba : LT.lt (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) a
    ⊢ LT.lt (↑(Nat.ceil a)) (HMul.hMul b a)
  -/
  obtain hab | hba := le_total a (b - 1)⁻¹
  · calc
      ⌈a⌉₊ ≤ (⌈(b - 1)⁻¹⌉₊ : α) := by gcongr
      _ < b * a := by rwa [← div_lt_iff₀']; positivity
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorSemiring α
      a b : α
      hb : LT.lt 1 b
      hba✝ : LT.lt (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      hba : LE.le (Inv.inv (HSub.hSub b 1)) a
      ⊢ LT.lt (↑(Nat.ceil a)) (HMul.hMul b a)
    -/
  · rw [← sub_pos] at hb
    calc
      ⌈a⌉₊ < a + 1 := ceil_lt_add_one <| hba.trans' <| by positivity
      _ = a + (b - 1) * (b - 1)⁻¹ := by rw [mul_inv_cancel₀]; positivity
      _ ≤ a + (b - 1) * a := by gcongr; positivity
      _ = b * a := by rw [sub_one_mul, add_sub_cancel]


lemma ceil_le_mul (hb : 1 < b) (hba : ⌈(b - 1)⁻¹⌉₊ / b ≤ a) : ⌈a⌉₊ ≤ b * a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorSemiring α
    a b : α
    hb : LT.lt 1 b
    hba : LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) a
    ⊢ LE.le (↑(Nat.ceil a)) (HMul.hMul b a)
  -/
  obtain rfl | hba := hba.eq_or_lt
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorSemiring α
      b : α
      hb : LT.lt 1 b
      hba : LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) (HDiv.hDiv ( …
      ⊢ LE.le (↑(Nat.ceil (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b))) (H …
    -/
  · rw [mul_div_cancel₀, cast_le, ceil_le]
      /-
        case inl
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorSemiring α
        b : α
        hb : LT.lt 1 b
        hba : LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) (HDiv.hDiv ( …
        ⊢ LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) ↑(Nat.ceil (Inv. …
      -/
    · exact _root_.div_le_self (by positivity) hb.le
      /-
        🎉 no goals
      -/
      /-
        case inl.hb
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorSemiring α
        b : α
        hb : LT.lt 1 b
        hba : LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) (HDiv.hDiv ( …
        ⊢ Ne b 0
      -/
    · positivity
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorSemiring α
      a b : α
      hb : LT.lt 1 b
      hba✝ : LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      hba : LT.lt (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      ⊢ LE.le (↑(Nat.ceil a)) (HMul.hMul b a)
    -/
  · exact (ceil_lt_mul hb hba).le
    /-
      🎉 no goals
    -/


lemma div_two_lt_floor (ha : 1 ≤ a) : a / 2 < ⌊a⌋₊ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorSemiring α
    a : α
    ha : LE.le 1 a
    ⊢ LT.lt (HDiv.hDiv a 2) ↑(Nat.floor a)
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  rw [div_eq_inv_mul]; refine mul_lt_floor ?_ ?_ ?_ <;> norm_num; assumption
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma ceil_lt_two_mul (ha : 2⁻¹ < a) : ⌈a⌉₊ < 2 * a :=
                             /-
                               α : Type u_2
                               inst✝¹ : LinearOrderedField α
                               inst✝ : FloorSemiring α
                               a : α
                               ha : LT.lt (Inv.inv 2) a
                               ⊢ LT.lt (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub 2 1)))) 2) a
                             -/
  ceil_lt_mul one_lt_two (by norm_num at ha ⊢; exact ha)
                                               /-
                                                 🎉 no goals
                                               -/


lemma ceil_le_two_mul (ha : 2⁻¹ ≤ a) : ⌈a⌉₊ ≤ 2 * a :=
                             /-
                               α : Type u_2
                               inst✝¹ : LinearOrderedField α
                               inst✝ : FloorSemiring α
                               a : α
                               ha : LE.le (Inv.inv 2) a
                               ⊢ LE.le (HDiv.hDiv (↑(Nat.ceil (Inv.inv (HSub.hSub 2 1)))) 2) a
                             -/
  ceil_le_mul one_lt_two (by norm_num at ha ⊢; exact ha)
                                               /-
                                                 🎉 no goals
                                               -/


/-- There exists at most one `FloorSemiring` structure on a linear ordered semiring. -/
theorem subsingleton_floorSemiring {α} [LinearOrderedSemiring α] :
    Subsingleton (FloorSemiring α) := by
  /-
    α : Type u_4
    inst✝ : LinearOrderedSemiring α
    ⊢ Subsingleton (FloorSemiring α)
  -/
  refine ⟨fun H₁ H₂ => ?_⟩
  /-
    α : Type u_4
    inst✝ : LinearOrderedSemiring α
    H₁ H₂ : FloorSemiring α
    ⊢ Eq H₁ H₂
  -/
  have : H₁.ceil = H₂.ceil := funext fun a => (H₁.gc_ceil.l_unique H₂.gc_ceil) fun n => rfl
  have : H₁.floor = H₂.floor := by
    ext a
    cases' lt_or_le a 0 with h h
    · rw [H₁.floor_of_neg, H₂.floor_of_neg] <;> exact h
    · refine eq_of_forall_le_iff fun n => ?_
      rw [H₁.gc_floor, H₂.gc_floor] <;> exact h
  /-
    α : Type u_4
    inst✝ : LinearOrderedSemiring α
    H₁ H₂ : FloorSemiring α
    this✝ : Eq FloorSemiring.ceil FloorSemiring.ceil
    this : Eq FloorSemiring.floor FloorSemiring.floor
    ⊢ Eq H₁ H₂
  -/
  cases H₁
  /-
    case mk
    α : Type u_4
    inst✝ : LinearOrderedSemiring α
    H₂ : FloorSemiring α
    floor✝ ceil✝ : α → Nat
    floor_of_neg✝ : ∀ {a : α}, LT.lt a 0 → Eq (floor✝ a) 0
    gc_floor✝ : ∀ {a : α} {n : Nat}, LE.le 0 a → Iff (LE.le n (floor✝ a)) (LE.le ( …
    gc_ceil✝ : GaloisConnection ceil✝ Nat.cast
    this✝ : Eq FloorSemiring.ceil FloorSemiring.ceil
    this : Eq FloorSemiring.floor FloorSemiring.floor
    ⊢ Eq { floor := floor✝, ceil := ceil✝, floor_of_neg := floor_of_neg✝, gc_floor …
  -/
  cases H₂
  /-
    case mk.mk
    α : Type u_4
    inst✝ : LinearOrderedSemiring α
    floor✝¹ ceil✝¹ : α → Nat
    floor_of_neg✝¹ : ∀ {a : α}, LT.lt a 0 → Eq (floor✝¹ a) 0
    gc_floor✝¹ : ∀ {a : α} {n : Nat}, LE.le 0 a → Iff (LE.le n (floor✝¹ a)) (LE.le …
    gc_ceil✝¹ : GaloisConnection ceil✝¹ Nat.cast
    floor✝ ceil✝ : α → Nat
    floor_of_neg✝ : ∀ {a : α}, LT.lt a 0 → Eq (floor✝ a) 0
    gc_floor✝ : ∀ {a : α} {n : Nat}, LE.le 0 a → Iff (LE.le n (floor✝ a)) (LE.le ( …
    gc_ceil✝ : GaloisConnection ceil✝ Nat.cast
    this✝ : Eq FloorSemiring.ceil FloorSemiring.ceil
    this : Eq FloorSemiring.floor FloorSemiring.floor
    ⊢ Eq { floor := floor✝¹, ceil := ceil✝¹, floor_of_neg := floor_of_neg✝¹, gc_fl …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- A `FloorRing` is a linear ordered ring over `α` with a function
`floor : α → ℤ` satisfying `∀ (z : ℤ) (a : α), z ≤ floor a ↔ (z : α) ≤ a)`.
-/
class FloorRing (α) [LinearOrderedRing α] where
  /-- `FloorRing.floor a` computes the greatest integer `z` such that `(z : α) ≤ a`. -/
  floor : α → ℤ
  /-- `FloorRing.ceil a` computes the least integer `z` such that `a ≤ (z : α)`. -/
  ceil : α → ℤ
  /-- `FloorRing.ceil` is the upper adjoint of the coercion `↑ : ℤ → α`. -/
  gc_coe_floor : GaloisConnection (↑) floor
  /-- `FloorRing.ceil` is the lower adjoint of the coercion `↑ : ℤ → α`. -/
  gc_ceil_coe : GaloisConnection ceil (↑)


instance : FloorRing ℤ where
  floor := id
  ceil := id
  gc_coe_floor a b := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a b : Int
      ⊢ Iff (LE.le (↑a) b) (LE.le a (id b))
    -/
    rw [Int.cast_id]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a b : Int
      ⊢ Iff (LE.le a b) (LE.le a (id b))
    -/
    rfl
    /-
      🎉 no goals
    -/
  gc_ceil_coe a b := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a b : Int
      ⊢ Iff (LE.le (id a) b) (LE.le a ↑b)
    -/
    rw [Int.cast_id]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      a b : Int
      ⊢ Iff (LE.le (id a) b) (LE.le a b)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A `FloorRing` constructor from the `floor` function alone. -/
def FloorRing.ofFloor (α) [LinearOrderedRing α] (floor : α → ℤ)
    (gc_coe_floor : GaloisConnection (↑) floor) : FloorRing α :=
  { floor
    ceil := fun a => -floor (-a)
    gc_coe_floor
                                 /-
                                   F : Type u_1
                                   α✝ : Type u_2
                                   β : Type u_3
                                   α : Type ?u.171087
                                   inst✝ : LinearOrderedRing α
                                   floor : α → Int
                                   gc_coe_floor : GaloisConnection Int.cast floor
                                   a : α
                                   z : Int
                                   ⊢ Iff (LE.le ((fun a => Neg.neg (floor (Neg.neg a))) a) z) (LE.le a ↑z)
                                 -/
    gc_ceil_coe := fun a z => by rw [neg_le, ← gc_coe_floor, Int.cast_neg, neg_le_neg_iff] }
                                 /-
                                   🎉 no goals
                                 -/


/-- A `FloorRing` constructor from the `ceil` function alone. -/
def FloorRing.ofCeil (α) [LinearOrderedRing α] (ceil : α → ℤ)
    (gc_ceil_coe : GaloisConnection ceil (↑)) : FloorRing α :=
  { floor := fun a => -ceil (-a)
    ceil
                                  /-
                                    F : Type u_1
                                    α✝ : Type u_2
                                    β : Type u_3
                                    α : Type ?u.172658
                                    inst✝ : LinearOrderedRing α
                                    ceil : α → Int
                                    gc_ceil_coe : GaloisConnection ceil Int.cast
                                    a : Int
                                    z : α
                                    ⊢ Iff (LE.le (↑a) z) (LE.le a ((fun a => Neg.neg (ceil (Neg.neg a))) z))
                                  -/
    gc_coe_floor := fun a z => by rw [le_neg, gc_ceil_coe, Int.cast_neg, neg_le_neg_iff]
                                  /-
                                    🎉 no goals
                                  -/
    gc_ceil_coe }


/-- `Int.floor a` is the greatest integer `z` such that `z ≤ a`. It is denoted with `⌊a⌋`. -/
def floor : α → ℤ :=
  FloorRing.floor


/-- `Int.ceil a` is the smallest integer `z` such that `a ≤ z`. It is denoted with `⌈a⌉`. -/
def ceil : α → ℤ :=
  FloorRing.ceil


/-- `Int.fract a`, the fractional part of `a`, is `a` minus its floor. -/
def fract (a : α) : α :=
  a - floor a


@[simp]
theorem floor_int : (Int.floor : ℤ → ℤ) = id :=
  rfl


@[simp]
theorem ceil_int : (Int.ceil : ℤ → ℤ) = id :=
  rfl


@[simp]
theorem fract_int : (Int.fract : ℤ → ℤ) = 0 :=
                     /-
                       x : Int
                       ⊢ Eq (Int.fract x) (0 x)
                     -/
  funext fun x => by simp [fract]
                     /-
                       🎉 no goals
                     -/


@[inherit_doc]
notation "⌊" a "⌋" => Int.floor a


@[inherit_doc]
notation "⌈" a "⌉" => Int.ceil a

-- Mathematical notation for `fract a` is usually `{a}`. Let's not even go there.


@[simp]
theorem floorRing_floor_eq : @FloorRing.floor = @Int.floor :=
  rfl


@[simp]
theorem floorRing_ceil_eq : @FloorRing.ceil = @Int.ceil :=
  rfl


theorem gc_coe_floor : GaloisConnection ((↑) : ℤ → α) floor :=
  FloorRing.gc_coe_floor


theorem le_floor : z ≤ ⌊a⌋ ↔ (z : α) ≤ a :=
  (gc_coe_floor z a).symm


theorem floor_lt : ⌊a⌋ < z ↔ a < z :=
  lt_iff_lt_of_le_iff_le le_floor


@[bound]
theorem floor_le (a : α) : (⌊a⌋ : α) ≤ a :=
  gc_coe_floor.l_u_le a


                                                 /-
                                                   α : Type u_2
                                                   inst✝¹ : LinearOrderedRing α
                                                   inst✝ : FloorRing α
                                                   z : Int
                                                   a : α
                                                   ⊢ Iff (LE.le (Int.floor a) z) (LT.lt a (HAdd.hAdd (↑z) 1))
                                                 -/
theorem floor_le_iff : ⌊a⌋ ≤ z ↔ a < z + 1 := by rw [← lt_add_one_iff, floor_lt]; norm_cast
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/

                                                 /-
                                                   α : Type u_2
                                                   inst✝¹ : LinearOrderedRing α
                                                   inst✝ : FloorRing α
                                                   z : Int
                                                   a : α
                                                   ⊢ Iff (LT.lt z (Int.floor a)) (LE.le (HAdd.hAdd (↑z) 1) a)
                                                 -/
theorem lt_floor_iff : z < ⌊a⌋ ↔ z + 1 ≤ a := by rw [← add_one_le_iff, le_floor]; norm_cast
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


                                             /-
                                               α : Type u_2
                                               inst✝¹ : LinearOrderedRing α
                                               inst✝ : FloorRing α
                                               a : α
                                               ⊢ Iff (LE.le 0 (Int.floor a)) (LE.le 0 a)
                                             -/
theorem floor_nonneg : 0 ≤ ⌊a⌋ ↔ 0 ≤ a := by rw [le_floor, Int.cast_zero]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                                         /-
                                                           α : Type u_2
                                                           inst✝¹ : LinearOrderedRing α
                                                           inst✝ : FloorRing α
                                                           z : Int
                                                           a : α
                                                           ⊢ Iff (LE.le (Int.floor a) (HSub.hSub z 1)) (LT.lt a ↑z)
                                                         -/
theorem floor_le_sub_one_iff : ⌊a⌋ ≤ z - 1 ↔ a < z := by rw [← floor_lt, le_sub_one_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem floor_le_neg_one_iff : ⌊a⌋ ≤ -1 ↔ a < 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Iff (LE.le (Int.floor a) (-1)) (LT.lt a 0)
  -/
  rw [← zero_sub (1 : ℤ), floor_le_sub_one_iff, cast_zero]
  /-
    🎉 no goals
  -/


@[bound]
theorem floor_nonpos (ha : a ≤ 0) : ⌊a⌋ ≤ 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le a 0
    ⊢ LE.le (Int.floor a) 0
  -/
  rw [← @cast_le α, Int.cast_zero]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le a 0
    ⊢ LE.le (↑(Int.floor a)) 0
  -/
  exact (floor_le a).trans ha
  /-
    🎉 no goals
  -/


theorem lt_succ_floor (a : α) : a < ⌊a⌋.succ :=
  floor_lt.1 <| Int.lt_succ_self _


@[simp, bound]
theorem lt_floor_add_one (a : α) : a < ⌊a⌋ + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
  -/
  simpa only [Int.succ, Int.cast_add, Int.cast_one] using lt_succ_floor a
  /-
    🎉 no goals
  -/


@[simp, bound]
theorem sub_one_lt_floor (a : α) : a - 1 < ⌊a⌋ :=
  sub_lt_iff_lt_add.2 (lt_floor_add_one a)


@[simp]
theorem floor_intCast (z : ℤ) : ⌊(z : α)⌋ = z :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    z a : Int
                                    ⊢ Iff (LE.le a (Int.floor ↑z)) (LE.le a z)
                                  -/
  eq_of_forall_le_iff fun a => by rw [le_floor, Int.cast_le]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem floor_natCast (n : ℕ) : ⌊(n : α)⌋ = n :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    n : Nat
                                    a : Int
                                    ⊢ Iff (LE.le a (Int.floor ↑n)) (LE.le a ↑n)
                                  -/
  eq_of_forall_le_iff fun a => by rw [le_floor, ← cast_natCast, cast_le]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝¹ : LinearOrderedRing α
                                           inst✝ : FloorRing α
                                           ⊢ Eq (Int.floor 0) 0
                                         -/
theorem floor_zero : ⌊(0 : α)⌋ = 0 := by rw [← cast_zero, floor_intCast]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                        /-
                                          α : Type u_2
                                          inst✝¹ : LinearOrderedRing α
                                          inst✝ : FloorRing α
                                          ⊢ Eq (Int.floor 1) 1
                                        -/
theorem floor_one : ⌊(1 : α)⌋ = 1 := by rw [← cast_one, floor_intCast]
                                        /-
                                          🎉 no goals
                                        -/


@[simp] theorem floor_ofNat (n : ℕ) [n.AtLeastTwo] : ⌊(ofNat(n) : α)⌋ = ofNat(n) :=
  floor_natCast n


@[mono]
theorem floor_mono : Monotone (floor : α → ℤ) :=
  gc_coe_floor.monotone_u


@[gcongr, bound] lemma floor_le_floor (hab : a ≤ b) : ⌊a⌋ ≤ ⌊b⌋ := floor_mono hab


theorem floor_pos : 0 < ⌊a⌋ ↔ 1 ≤ a := by
  -- Porting note: broken `convert le_floor`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Iff (LT.lt 0 (Int.floor a)) (LE.le 1 a)
  -/
  rw [Int.lt_iff_add_one_le, zero_add, le_floor, cast_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem floor_add_int (a : α) (z : ℤ) : ⌊a + z⌋ = ⌊a⌋ + z :=
  eq_of_forall_le_iff fun a => by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a✝ : α
      z a : Int
      ⊢ Iff (LE.le a (Int.floor (HAdd.hAdd a✝ ↑z))) (LE.le a (HAdd.hAdd (Int.floor a …
    -/
    rw [le_floor, ← sub_le_iff_le_add, ← sub_le_iff_le_add, le_floor, Int.cast_sub]
    /-
      🎉 no goals
    -/


@[simp]
theorem floor_add_one (a : α) : ⌊a + 1⌋ = ⌊a⌋ + 1 := by
  -- Porting note: broken `convert floor_add_int a 1`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Int.floor (HAdd.hAdd a 1)) (HAdd.hAdd (Int.floor a) 1)
  -/
  rw [← cast_one, floor_add_int]
  /-
    🎉 no goals
  -/


@[bound]
theorem le_floor_add (a b : α) : ⌊a⌋ + ⌊b⌋ ≤ ⌊a + b⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd (Int.floor a) (Int.floor b)) (Int.floor (HAdd.hAdd a b))
  -/
  rw [le_floor, Int.cast_add]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd ↑(Int.floor a) ↑(Int.floor b)) (HAdd.hAdd a b)
  -/
             /-
               🎉 no goals
             -/
  gcongr <;> apply floor_le
             /-
               🎉 no goals
             -/


@[bound]
theorem le_floor_add_floor (a b : α) : ⌊a + b⌋ - 1 ≤ ⌊a⌋ + ⌊b⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HSub.hSub (Int.floor (HAdd.hAdd a b)) 1) (HAdd.hAdd (Int.floor a) (In …
  -/
  rw [← sub_le_iff_le_add, le_floor, Int.cast_sub, sub_le_comm, Int.cast_sub, Int.cast_one]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HSub.hSub (HSub.hSub (↑(Int.floor (HAdd.hAdd a b))) 1) a) ↑(Int.floor …
  -/
  refine le_trans ?_ (sub_one_lt_floor _).le
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HSub.hSub (HSub.hSub (↑(Int.floor (HAdd.hAdd a b))) 1) a) (HSub.hSub  …
  -/
  rw [sub_le_iff_le_add', ← add_sub_assoc, sub_le_sub_iff_right]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (↑(Int.floor (HAdd.hAdd a b))) (HAdd.hAdd a b)
  -/
  exact floor_le _
  /-
    🎉 no goals
  -/


@[simp]
theorem floor_int_add (z : ℤ) (a : α) : ⌊↑z + a⌋ = z + ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    z : Int
    a : α
    ⊢ Eq (Int.floor (HAdd.hAdd (↑z) a)) (HAdd.hAdd z (Int.floor a))
  -/
  simpa only [add_comm] using floor_add_int a z
  /-
    🎉 no goals
  -/


@[simp]
theorem floor_add_nat (a : α) (n : ℕ) : ⌊a + n⌋ = ⌊a⌋ + n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (Int.floor (HAdd.hAdd a ↑n)) (HAdd.hAdd (Int.floor a) ↑n)
  -/
  rw [← Int.cast_natCast, floor_add_int]
  /-
    🎉 no goals
  -/


@[simp]
theorem floor_add_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌊a + ofNat(n)⌋ = ⌊a⌋ + ofNat(n) :=
  floor_add_nat a n


@[simp]
theorem floor_nat_add (n : ℕ) (a : α) : ⌊↑n + a⌋ = n + ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    n : Nat
    a : α
    ⊢ Eq (Int.floor (HAdd.hAdd (↑n) a)) (HAdd.hAdd (↑n) (Int.floor a))
  -/
  rw [← Int.cast_natCast, floor_int_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem floor_ofNat_add (n : ℕ) [n.AtLeastTwo] (a : α) :
    ⌊ofNat(n) + a⌋ = ofNat(n) + ⌊a⌋ :=
  floor_nat_add n a


@[simp]
theorem floor_sub_int (a : α) (z : ℤ) : ⌊a - z⌋ = ⌊a⌋ - z :=
               /-
                 α : Type u_2
                 inst✝¹ : LinearOrderedRing α
                 inst✝ : FloorRing α
                 a : α
                 z : Int
                 ⊢ Eq (Int.floor (HSub.hSub a ↑z)) (Int.floor (HAdd.hAdd a ↑(Neg.neg z)))
               -/
  Eq.trans (by rw [Int.cast_neg, sub_eq_add_neg]) (floor_add_int _ _)
               /-
                 🎉 no goals
               -/


@[simp]
theorem floor_sub_nat (a : α) (n : ℕ) : ⌊a - n⌋ = ⌊a⌋ - n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (Int.floor (HSub.hSub a ↑n)) (HSub.hSub (Int.floor a) ↑n)
  -/
  rw [← Int.cast_natCast, floor_sub_int]
  /-
    🎉 no goals
  -/


@[simp] theorem floor_sub_one (a : α) : ⌊a - 1⌋ = ⌊a⌋ - 1 := mod_cast floor_sub_nat a 1


@[simp]
theorem floor_sub_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌊a - ofNat(n)⌋ = ⌊a⌋ - ofNat(n) :=
  floor_sub_nat a n


theorem abs_sub_lt_one_of_floor_eq_floor {α : Type*} [LinearOrderedCommRing α] [FloorRing α]
    {a b : α} (h : ⌊a⌋ = ⌊b⌋) : |a - b| < 1 := by
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  have : a < ⌊a⌋ + 1 := lt_floor_add_one a
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    this : LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  have : b < ⌊b⌋ + 1 := lt_floor_add_one b
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    this✝ : LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
    this : LT.lt b (HAdd.hAdd (↑(Int.floor b)) 1)
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  have : (⌊a⌋ : α) = ⌊b⌋ := Int.cast_inj.2 h
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    this✝¹ : LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
    this✝ : LT.lt b (HAdd.hAdd (↑(Int.floor b)) 1)
    this : Eq ↑(Int.floor a) ↑(Int.floor b)
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  have : (⌊a⌋ : α) ≤ a := floor_le a
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    this✝² : LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
    this✝¹ : LT.lt b (HAdd.hAdd (↑(Int.floor b)) 1)
    this✝ : Eq ↑(Int.floor a) ↑(Int.floor b)
    this : LE.le (↑(Int.floor a)) a
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  have : (⌊b⌋ : α) ≤ b := floor_le b
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedCommRing α
    inst✝ : FloorRing α
    a b : α
    h : Eq (Int.floor a) (Int.floor b)
    this✝³ : LT.lt a (HAdd.hAdd (↑(Int.floor a)) 1)
    this✝² : LT.lt b (HAdd.hAdd (↑(Int.floor b)) 1)
    this✝¹ : Eq ↑(Int.floor a) ↑(Int.floor b)
    this✝ : LE.le (↑(Int.floor a)) a
    this : LE.le (↑(Int.floor b)) b
    ⊢ LT.lt (abs (HSub.hSub a b)) 1
  -/
  exact abs_sub_lt_iff.2 ⟨by linarith, by linarith⟩
  /-
    🎉 no goals
  -/


theorem floor_eq_iff : ⌊a⌋ = z ↔ ↑z ≤ a ∧ a < z + 1 := by
  rw [le_antisymm_iff, le_floor, ← Int.lt_add_one_iff, floor_lt, Int.cast_add, Int.cast_one,
    and_comm]


@[simp]
                                                              /-
                                                                α : Type u_2
                                                                inst✝¹ : LinearOrderedRing α
                                                                inst✝ : FloorRing α
                                                                a : α
                                                                ⊢ Iff (Eq (Int.floor a) 0) (Membership.mem (Set.Ico 0 1) a)
                                                              -/
theorem floor_eq_zero_iff : ⌊a⌋ = 0 ↔ a ∈ Ico (0 : α) 1 := by simp [floor_eq_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem floor_eq_on_Ico (n : ℤ) : ∀ a ∈ Set.Ico (n : α) (n + 1), ⌊a⌋ = n := fun _ ⟨h₀, h₁⟩ =>
  floor_eq_iff.mpr ⟨h₀, h₁⟩


theorem floor_eq_on_Ico' (n : ℤ) : ∀ a ∈ Set.Ico (n : α) (n + 1), (⌊a⌋ : α) = n := fun a ha =>
  congr_arg _ <| floor_eq_on_Ico n a ha

-- Porting note: in mathlib3 there was no need for the type annotation in `(m:α)`

@[simp]
theorem preimage_floor_singleton (m : ℤ) : (floor : α → ℤ) ⁻¹' {m} = Ico (m : α) (m + 1) :=
  ext fun _ => floor_eq_iff


@[simp]
theorem self_sub_floor (a : α) : a - ⌊a⌋ = fract a :=
  rfl


@[simp]
theorem floor_add_fract (a : α) : (⌊a⌋ : α) + fract a = a :=
  add_sub_cancel _ _


@[simp]
theorem fract_add_floor (a : α) : fract a + ⌊a⌋ = a :=
  sub_add_cancel _ _


@[simp]
theorem fract_add_int (a : α) (m : ℤ) : fract (a + m) = fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Int
    ⊢ Eq (Int.fract (HAdd.hAdd a ↑m)) (Int.fract a)
  -/
  rw [fract]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Int
    ⊢ Eq (HSub.hSub (HAdd.hAdd a ↑m) ↑(Int.floor (HAdd.hAdd a ↑m))) (Int.fract a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_add_nat (a : α) (m : ℕ) : fract (a + m) = fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Nat
    ⊢ Eq (Int.fract (HAdd.hAdd a ↑m)) (Int.fract a)
  -/
  rw [fract]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Nat
    ⊢ Eq (HSub.hSub (HAdd.hAdd a ↑m) ↑(Int.floor (HAdd.hAdd a ↑m))) (Int.fract a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_add_one (a : α) : fract (a + 1) = fract a := mod_cast fract_add_nat a 1


@[simp]
theorem fract_add_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    fract (a + ofNat(n)) = fract a :=
  fract_add_nat a n


@[simp]
                                                                       /-
                                                                         α : Type u_2
                                                                         inst✝¹ : LinearOrderedRing α
                                                                         inst✝ : FloorRing α
                                                                         m : Int
                                                                         a : α
                                                                         ⊢ Eq (Int.fract (HAdd.hAdd (↑m) a)) (Int.fract a)
                                                                       -/
theorem fract_int_add (m : ℤ) (a : α) : fract (↑m + a) = fract a := by rw [add_comm, fract_add_int]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                       /-
                                                                         α : Type u_2
                                                                         inst✝¹ : LinearOrderedRing α
                                                                         inst✝ : FloorRing α
                                                                         n : Nat
                                                                         a : α
                                                                         ⊢ Eq (Int.fract (HAdd.hAdd (↑n) a)) (Int.fract a)
                                                                       -/
theorem fract_nat_add (n : ℕ) (a : α) : fract (↑n + a) = fract a := by rw [add_comm, fract_add_nat]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem fract_one_add (a : α) : fract (1 + a) = fract a := mod_cast fract_nat_add 1 a


@[simp]
theorem fract_ofNat_add (n : ℕ) [n.AtLeastTwo] (a : α) :
    fract (ofNat(n) + a) = fract a :=
  fract_nat_add n a


@[simp]
theorem fract_sub_int (a : α) (m : ℤ) : fract (a - m) = fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Int
    ⊢ Eq (Int.fract (HSub.hSub a ↑m)) (Int.fract a)
  -/
  rw [fract]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    m : Int
    ⊢ Eq (HSub.hSub (HSub.hSub a ↑m) ↑(Int.floor (HSub.hSub a ↑m))) (Int.fract a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_sub_nat (a : α) (n : ℕ) : fract (a - n) = fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (Int.fract (HSub.hSub a ↑n)) (Int.fract a)
  -/
  rw [fract]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (HSub.hSub (HSub.hSub a ↑n) ↑(Int.floor (HSub.hSub a ↑n))) (Int.fract a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_sub_one (a : α) : fract (a - 1) = fract a := mod_cast fract_sub_nat a 1


@[simp]
theorem fract_sub_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    fract (a - ofNat(n)) = fract a :=
  fract_sub_nat a n

-- Was a duplicate lemma under a bad name


theorem fract_add_le (a b : α) : fract (a + b) ≤ fract a + fract b := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (Int.fract (HAdd.hAdd a b)) (HAdd.hAdd (Int.fract a) (Int.fract b))
  -/
  rw [fract, fract, fract, sub_add_sub_comm, sub_le_sub_iff_left, ← Int.cast_add, Int.cast_le]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd (Int.floor a) (Int.floor b)) (Int.floor (HAdd.hAdd a b))
  -/
  exact le_floor_add _ _
  /-
    🎉 no goals
  -/


theorem fract_add_fract_le (a b : α) : fract a + fract b ≤ fract (a + b) + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd (Int.fract a) (Int.fract b)) (HAdd.hAdd (Int.fract (HAdd.hA …
  -/
  rw [fract, fract, fract, sub_add_sub_comm, sub_add, sub_le_sub_iff_left]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HSub.hSub (↑(Int.floor (HAdd.hAdd a b))) 1) (HAdd.hAdd ↑(Int.floor a) …
  -/
  exact mod_cast le_floor_add_floor a b
  /-
    🎉 no goals
  -/


@[simp]
theorem self_sub_fract (a : α) : a - fract a = ⌊a⌋ :=
  sub_sub_cancel _ _


@[simp]
theorem fract_sub_self (a : α) : fract a - a = -⌊a⌋ :=
  sub_sub_cancel_left _ _


@[simp]
theorem fract_nonneg (a : α) : 0 ≤ fract a :=
  sub_nonneg.2 <| floor_le _


/-- The fractional part of `a` is positive if and only if `a ≠ ⌊a⌋`. -/
lemma fract_pos : 0 < fract a ↔ a ≠ ⌊a⌋ :=
  (fract_nonneg a).lt_iff_ne.trans <| ne_comm.trans sub_ne_zero


theorem fract_lt_one (a : α) : fract a < 1 :=
  sub_lt_comm.1 <| sub_one_lt_floor _


@[simp]
                                             /-
                                               α : Type u_2
                                               inst✝¹ : LinearOrderedRing α
                                               inst✝ : FloorRing α
                                               ⊢ Eq (Int.fract 0) 0
                                             -/
theorem fract_zero : fract (0 : α) = 0 := by rw [fract, floor_zero, cast_zero, sub_self]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                            /-
                                              α : Type u_2
                                              inst✝¹ : LinearOrderedRing α
                                              inst✝ : FloorRing α
                                              ⊢ Eq (Int.fract 1) 0
                                            -/
theorem fract_one : fract (1 : α) = 0 := by simp [fract]
                                            /-
                                              🎉 no goals
                                            -/


theorem abs_fract : |fract a| = fract a :=
  abs_eq_self.mpr <| fract_nonneg a


@[simp]
theorem abs_one_sub_fract : |1 - fract a| = 1 - fract a :=
  abs_eq_self.mpr <| sub_nonneg.mpr (fract_lt_one a).le


@[simp]
theorem fract_intCast (z : ℤ) : fract (z : α) = 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    z : Int
    ⊢ Eq (Int.fract ↑z) 0
  -/
  unfold fract
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    z : Int
    ⊢ Eq (HSub.hSub ↑z ↑(Int.floor ↑z)) 0
  -/
  rw [floor_intCast]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    z : Int
    ⊢ Eq (HSub.hSub ↑z ↑z) 0
  -/
  exact sub_self _
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : LinearOrderedRing α
                                                          inst✝ : FloorRing α
                                                          n : Nat
                                                          ⊢ Eq (Int.fract ↑n) 0
                                                        -/
theorem fract_natCast (n : ℕ) : fract (n : α) = 0 := by simp [fract]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem fract_ofNat (n : ℕ) [n.AtLeastTwo] :
    fract (ofNat(n) : α) = 0 :=
  fract_natCast n


theorem fract_floor (a : α) : fract (⌊a⌋ : α) = 0 :=
  fract_intCast _


@[simp]
theorem floor_fract (a : α) : ⌊fract a⌋ = 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Int.floor (Int.fract a)) 0
  -/
  rw [floor_eq_iff, Int.cast_zero, zero_add]; exact ⟨fract_nonneg _, fract_lt_one _⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem fract_eq_iff {a b : α} : fract a = b ↔ 0 ≤ b ∧ b < 1 ∧ ∃ z : ℤ, a - b = z :=
  ⟨fun h => by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      h : Eq (Int.fract a) b
      ⊢ And (LE.le 0 b) (And (LT.lt b 1) (Exists fun z => Eq (HSub.hSub a b) ↑z))
    -/
    rw [← h]
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      h : Eq (Int.fract a) b
      ⊢ And (LE.le 0 (Int.fract a)) (And (LT.lt (Int.fract a) 1) (Exists fun z => Eq …
    -/
    exact ⟨fract_nonneg _, fract_lt_one _, ⟨⌊a⌋, sub_sub_cancel _ _⟩⟩,
    /-
      🎉 no goals
    -/
   by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      ⊢ And (LE.le 0 b) (And (LT.lt b 1) (Exists fun z => Eq (HSub.hSub a b) ↑z)) →  …
    -/
    rintro ⟨h₀, h₁, z, hz⟩
    rw [← self_sub_floor, eq_comm, eq_sub_iff_add_eq, add_comm, ← eq_sub_iff_add_eq, hz,
      Int.cast_inj, floor_eq_iff, ← hz]
    /-
      case intro.intro.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      h₀ : LE.le 0 b
      h₁ : LT.lt b 1
      z : Int
      hz : Eq (HSub.hSub a b) ↑z
      ⊢ And (LE.le (HSub.hSub a b) a) (LT.lt a (HAdd.hAdd (HSub.hSub a b) 1))
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> simpa [sub_eq_add_neg, add_assoc] ⟩
                    /-
                      🎉 no goals
                    -/


theorem fract_eq_fract {a b : α} : fract a = fract b ↔ ∃ z : ℤ, a - b = z :=
                           /-
                             α : Type u_2
                             inst✝¹ : LinearOrderedRing α
                             inst✝ : FloorRing α
                             a b : α
                             h : Eq (Int.fract a) (Int.fract b)
                             ⊢ Eq (HSub.hSub a b) ↑(HSub.hSub (Int.floor a) (Int.floor b))
                           -/
  ⟨fun h => ⟨⌊a⌋ - ⌊b⌋, by unfold fract at h; rw [Int.cast_sub, sub_eq_sub_iff_sub_eq_sub.1 h]⟩,
                                              /-
                                                🎉 no goals
                                              -/
   by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      ⊢ (Exists fun z => Eq (HSub.hSub a b) ↑z) → Eq (Int.fract a) (Int.fract b)
    -/
    rintro ⟨z, hz⟩
    /-
      case intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      z : Int
      hz : Eq (HSub.hSub a b) ↑z
      ⊢ Eq (Int.fract a) (Int.fract b)
    -/
    refine fract_eq_iff.2 ⟨fract_nonneg _, fract_lt_one _, z + ⌊b⌋, ?_⟩
    /-
      case intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      z : Int
      hz : Eq (HSub.hSub a b) ↑z
      ⊢ Eq (HSub.hSub a (Int.fract b)) ↑(HAdd.hAdd z (Int.floor b))
    -/
    rw [eq_add_of_sub_eq hz, add_comm, Int.cast_add]
    /-
      case intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      z : Int
      hz : Eq (HSub.hSub a b) ↑z
      ⊢ Eq (HSub.hSub (HAdd.hAdd b ↑z) (Int.fract b)) (HAdd.hAdd ↑z ↑(Int.floor b))
    -/
    exact add_sub_sub_cancel _ _ _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem fract_eq_self {a : α} : fract a = a ↔ 0 ≤ a ∧ a < 1 :=
                                                                    /-
                                                                      α : Type u_2
                                                                      inst✝¹ : LinearOrderedRing α
                                                                      inst✝ : FloorRing α
                                                                      a : α
                                                                      ⊢ Eq (HSub.hSub a a) ↑0
                                                                    -/
  fract_eq_iff.trans <| and_assoc.symm.trans <| and_iff_left ⟨0, by simp⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem fract_fract (a : α) : fract (fract a) = fract a :=
  fract_eq_self.2 ⟨fract_nonneg _, fract_lt_one _⟩


theorem fract_add (a b : α) : ∃ z : ℤ, fract (a + b) - fract a - fract b = z :=
  ⟨⌊a⌋ + ⌊b⌋ - ⌊a + b⌋, by
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      ⊢ Eq (HSub.hSub (HSub.hSub (Int.fract (HAdd.hAdd a b)) (Int.fract a)) (Int.fra …
    -/
    unfold fract
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd a b) ↑(Int.floor (HAdd.hAdd a …
    -/
    simp only [sub_eq_add_neg, neg_add_rev, neg_neg, cast_add, cast_neg]
    /-
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a b : α
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd a b) (Neg.neg ↑(Int.floor (HA …
    -/
    /-
      🎉 no goals
    -/
    abel⟩
    /-
      🎉 no goals
    -/


theorem fract_neg {x : α} (hx : fract x ≠ 0) : fract (-x) = 1 - fract x := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ Eq (Int.fract (Neg.neg x)) (HSub.hSub 1 (Int.fract x))
  -/
  rw [fract_eq_iff]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ And (LE.le 0 (HSub.hSub 1 (Int.fract x))) (And (LT.lt (HSub.hSub 1 (Int.frac …
  -/
  constructor
    /-
      case left
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      hx : Ne (Int.fract x) 0
      ⊢ LE.le 0 (HSub.hSub 1 (Int.fract x))
    -/
  · rw [le_sub_iff_add_le, zero_add]
    /-
      case left
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      hx : Ne (Int.fract x) 0
      ⊢ LE.le (Int.fract x) 1
    -/
    exact (fract_lt_one x).le
    /-
      🎉 no goals
    -/
  /-
    case right
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ And (LT.lt (HSub.hSub 1 (Int.fract x)) 1) (Exists fun z => Eq (HSub.hSub (Ne …
  -/
  refine ⟨sub_lt_self _ (lt_of_le_of_ne' (fract_nonneg x) hx), -⌊x⌋ - 1, ?_⟩
  /-
    case right
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ Eq (HSub.hSub (Neg.neg x) (HSub.hSub 1 (Int.fract x))) ↑(HSub.hSub (Neg.neg  …
  -/
  simp only [sub_sub_eq_add_sub, cast_sub, cast_neg, cast_one, sub_left_inj]
  /-
    case right
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ Eq (HAdd.hAdd (Neg.neg x) (Int.fract x)) (Neg.neg ↑(Int.floor x))
  -/
  conv in -x => rw [← floor_add_fract x]
  /-
    case right
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    hx : Ne (Int.fract x) 0
    ⊢ Eq (HAdd.hAdd (Neg.neg (HAdd.hAdd (↑(Int.floor x)) (Int.fract x))) (Int.frac …
  -/
  simp [-floor_add_fract]
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_neg_eq_zero {x : α} : fract (-x) = 0 ↔ fract x = 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    ⊢ Iff (Eq (Int.fract (Neg.neg x)) 0) (Eq (Int.fract x) 0)
  -/
  simp only [fract_eq_iff, le_refl, zero_lt_one, tsub_zero, true_and]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    ⊢ Iff (Exists fun z => Eq (Neg.neg x) ↑z) (Exists fun z => Eq x ↑z)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  constructor <;> rintro ⟨z, hz⟩ <;> use -z <;> simp [← hz]
                                                /-
                                                  🎉 no goals
                                                -/


theorem fract_mul_nat (a : α) (b : ℕ) : ∃ z : ℤ, fract a * b - fract (a * b) = z := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    b : Nat
    ⊢ Exists fun z => Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑b) (Int.fract (HMul. …
  -/
  induction' b with c hc
    /-
      case zero
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      ⊢ Exists fun z => Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑0) (Int.fract (HMul. …
    -/
  · use 0; simp
           /-
             🎉 no goals
           -/
    /-
      case succ
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      hc : Exists fun z => Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HM …
      ⊢ Exists fun z => Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑(HAdd.hAdd c 1)) (In …
    -/
  · rcases hc with ⟨z, hz⟩
    /-
      case succ.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      z : Int
      hz : Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HMul.hMul a ↑c))) ↑z
      ⊢ Exists fun z => Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑(HAdd.hAdd c 1)) (In …
    -/
    rw [Nat.cast_add, mul_add, mul_add, Nat.cast_one, mul_one, mul_one]
    /-
      case succ.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      z : Int
      hz : Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HMul.hMul a ↑c))) ↑z
      ⊢ Exists fun z => Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Int.fract a) ↑c) (Int.f …
    -/
    rcases fract_add (a * c) a with ⟨y, hy⟩
    /-
      case succ.intro.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      z : Int
      hz : Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HMul.hMul a ↑c))) ↑z
      y : Int
      hy : Eq (HSub.hSub (HSub.hSub (Int.fract (HAdd.hAdd (HMul.hMul a ↑c) a)) (Int. …
      ⊢ Exists fun z => Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Int.fract a) ↑c) (Int.f …
    -/
    use z - y
    /-
      case h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      z : Int
      hz : Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HMul.hMul a ↑c))) ↑z
      y : Int
      hy : Eq (HSub.hSub (HSub.hSub (Int.fract (HAdd.hAdd (HMul.hMul a ↑c) a)) (Int. …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Int.fract a) ↑c) (Int.fract a)) (Int.fr …
    -/
    rw [Int.cast_sub, ← hz, ← hy]
    /-
      case h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      c : Nat
      z : Int
      hz : Eq (HSub.hSub (HMul.hMul (Int.fract a) ↑c) (Int.fract (HMul.hMul a ↑c))) ↑z
      y : Int
      hy : Eq (HSub.hSub (HSub.hSub (Int.fract (HAdd.hAdd (HMul.hMul a ↑c) a)) (Int. …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Int.fract a) ↑c) (Int.fract a)) (Int.fr …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/

-- Porting note: in mathlib3 there was no need for the type annotation in `(m:α)`

theorem preimage_fract (s : Set α) :
    fract ⁻¹' s = ⋃ m : ℤ, (fun x => x - (m : α)) ⁻¹' (s ∩ Ico (0 : α) 1) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    ⊢ Eq (Set.preimage Int.fract s) (Set.iUnion fun m => Set.preimage (fun x => HS …
  -/
  ext x
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (Set.preimage Int.fract s) x) (Membership.mem (Set.iUnio …
  -/
  simp only [mem_preimage, mem_iUnion, mem_inter_iff]
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem s (Int.fract x)) (Exists fun i => And (Membership.mem s  …
  -/
  refine ⟨fun h => ⟨⌊x⌋, h, fract_nonneg x, fract_lt_one x⟩, ?_⟩
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    ⊢ (Exists fun i => And (Membership.mem s (HSub.hSub x ↑i)) (Membership.mem (Se …
  -/
  rintro ⟨m, hms, hm0, hm1⟩
  /-
    case h.intro.intro.intro
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    m : Int
    hms : Membership.mem s (HSub.hSub x ↑m)
    hm0 : LE.le 0 (HSub.hSub x ↑m)
    hm1 : LT.lt (HSub.hSub x ↑m) 1
    ⊢ Membership.mem s (Int.fract x)
  -/
  obtain rfl : ⌊x⌋ = m := floor_eq_iff.2 ⟨sub_nonneg.1 hm0, sub_lt_iff_lt_add'.1 hm1⟩
  /-
    case h.intro.intro.intro
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    hms : Membership.mem s (HSub.hSub x ↑(Int.floor x))
    hm0 : LE.le 0 (HSub.hSub x ↑(Int.floor x))
    hm1 : LT.lt (HSub.hSub x ↑(Int.floor x)) 1
    ⊢ Membership.mem s (Int.fract x)
  -/
  exact hms
  /-
    🎉 no goals
  -/


theorem image_fract (s : Set α) : fract '' s = ⋃ m : ℤ, (fun x : α => x - m) '' s ∩ Ico 0 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    ⊢ Eq (Set.image Int.fract s) (Set.iUnion fun m => Inter.inter (Set.image (fun  …
  -/
  ext x
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (Set.image Int.fract s) x) (Membership.mem (Set.iUnion f …
  -/
  simp only [mem_image, mem_inter_iff, mem_iUnion]; constructor
    /-
      case h.mp
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      s : Set α
      x : α
      ⊢ (Exists fun x_1 => And (Membership.mem s x_1) (Eq (Int.fract x_1) x)) → Exis …
    -/
  · rintro ⟨y, hy, rfl⟩
    /-
      case h.mp.intro.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      s : Set α
      y : α
      hy : Membership.mem s y
      ⊢ Exists fun i => And (Exists fun x => And (Membership.mem s x) (Eq (HSub.hSub …
    -/
    exact ⟨⌊y⌋, ⟨y, hy, rfl⟩, fract_nonneg y, fract_lt_one y⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      s : Set α
      x : α
      ⊢ (Exists fun i => And (Exists fun x_1 => And (Membership.mem s x_1) (Eq (HSub …
    -/
  · rintro ⟨m, ⟨y, hys, rfl⟩, h0, h1⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      s : Set α
      m : Int
      y : α
      hys : Membership.mem s y
      h0 : LE.le 0 (HSub.hSub y ↑m)
      h1 : LT.lt (HSub.hSub y ↑m) 1
      ⊢ Exists fun x => And (Membership.mem s x) (Eq (Int.fract x) (HSub.hSub y ↑m))
    -/
    obtain rfl : ⌊y⌋ = m := floor_eq_iff.2 ⟨sub_nonneg.1 h0, sub_lt_iff_lt_add'.1 h1⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      s : Set α
      y : α
      hys : Membership.mem s y
      h0 : LE.le 0 (HSub.hSub y ↑(Int.floor y))
      h1 : LT.lt (HSub.hSub y ↑(Int.floor y)) 1
      ⊢ Exists fun x => And (Membership.mem s x) (Eq (Int.fract x) (HSub.hSub y ↑(In …
    -/
    exact ⟨y, hys, rfl⟩
    /-
      🎉 no goals
    -/


theorem fract_div_mul_self_mem_Ico (a b : k) (ha : 0 < a) : fract (b / a) * a ∈ Ico 0 a :=
  ⟨(mul_nonneg_iff_of_pos_right ha).2 (fract_nonneg (b / a)),
    (mul_lt_iff_lt_one_left ha).2 (fract_lt_one (b / a))⟩


theorem fract_div_mul_self_add_zsmul_eq (a b : k) (ha : a ≠ 0) :
    fract (b / a) * a + ⌊b / a⌋ • a = b := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a b : k
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Int.fract (HDiv.hDiv b a)) a) (HSMul.hSMul (Int.fl …
  -/
  rw [zsmul_eq_mul, ← add_mul, fract_add_floor, div_mul_cancel₀ b ha]
  /-
    🎉 no goals
  -/


theorem sub_floor_div_mul_nonneg (a : k) (hb : 0 < b) : 0 ≤ a - ⌊a / b⌋ * b :=
  sub_nonneg_of_le <| (le_div_iff₀ hb).1 <| floor_le _


theorem sub_floor_div_mul_lt (a : k) (hb : 0 < b) : a - ⌊a / b⌋ * b < b :=
  sub_lt_iff_lt_add.2 <| by
    -- Porting note: `← one_add_mul` worked in mathlib3 without the argument
    /-
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      b a : k
      hb : LT.lt 0 b
      ⊢ LT.lt a (HAdd.hAdd b (HMul.hMul (↑(Int.floor (HDiv.hDiv a b))) b))
    -/
    rw [← one_add_mul _ b, ← div_lt_iff₀ hb, add_comm]
    /-
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      b a : k
      hb : LT.lt 0 b
      ⊢ LT.lt (HDiv.hDiv a b) (HAdd.hAdd (↑(Int.floor (HDiv.hDiv a b))) 1)
    -/
    exact lt_floor_add_one _
    /-
      🎉 no goals
    -/


theorem fract_div_natCast_eq_div_natCast_mod {m n : ℕ} : fract ((m : k) / n) = ↑(m % n) / n := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    m n : Nat
    ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑(HMod.hMod m n) ↑n)
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m : Nat
      ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑0)) (HDiv.hDiv ↑(HMod.hMod m 0) ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  have hn' : 0 < (n : k) := by
    norm_cast
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    m n : Nat
    hn : GT.gt n 0
    hn' : LT.lt 0 ↑n
    ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑(HMod.hMod m n) ↑n)
  -/
  refine fract_eq_iff.mpr ⟨?_, ?_, m / n, ?_⟩
    /-
      case inr.refine_1
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m n : Nat
      hn : GT.gt n 0
      hn' : LT.lt 0 ↑n
      ⊢ LE.le 0 (HDiv.hDiv ↑(HMod.hMod m n) ↑n)
    -/
  · positivity
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m n : Nat
      hn : GT.gt n 0
      hn' : LT.lt 0 ↑n
      ⊢ LT.lt (HDiv.hDiv ↑(HMod.hMod m n) ↑n) 1
    -/
  · simpa only [div_lt_one hn', Nat.cast_lt] using m.mod_lt hn
    /-
      🎉 no goals
    -/
  · rw [sub_eq_iff_eq_add', ← mul_right_inj' hn'.ne', mul_div_cancel₀ _ hn'.ne', mul_add,
      mul_div_cancel₀ _ hn'.ne']
    /-
      case inr.refine_3
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m n : Nat
      hn : GT.gt n 0
      hn' : LT.lt 0 ↑n
      ⊢ Eq (↑m) (HAdd.hAdd (↑(HMod.hMod m n)) (HMul.hMul ↑n ↑(HDiv.hDiv ↑m ↑n)))
    -/
    norm_cast
    /-
      case inr.refine_3
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m n : Nat
      hn : GT.gt n 0
      hn' : LT.lt 0 ↑n
      ⊢ Eq (↑m) (HAdd.hAdd ↑(HMod.hMod m n) ↑(HMul.hMul n (HDiv.hDiv m n)))
    -/
    rw [← Nat.cast_add, Nat.mod_add_div m n]
    /-
      🎉 no goals
    -/


theorem fract_div_intCast_eq_div_intCast_mod {m : ℤ} {n : ℕ} :
    fract ((m : k) / n) = ↑(m % n) / n := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    m : Int
    n : Nat
    ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑(HMod.hMod m ↑n) ↑n)
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      m : Int
      ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑0)) (HDiv.hDiv ↑(HMod.hMod m ↑0) ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    m : Int
    n : Nat
    hn : GT.gt n 0
    ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑(HMod.hMod m ↑n) ↑n)
  -/
  replace hn : 0 < (n : k) := by norm_cast
  have : ∀ {l : ℤ}, 0 ≤ l → fract ((l : k) / n) = ↑(l % n) / n := by
    intros l hl
    obtain ⟨l₀, rfl | rfl⟩ := l.eq_nat_or_neg
    · rw [cast_natCast, ← natCast_mod, cast_natCast, fract_div_natCast_eq_div_natCast_mod]
    · rw [Right.nonneg_neg_iff, natCast_nonpos_iff] at hl
      simp [hl]
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    m : Int
    n : Nat
    hn : LT.lt 0 ↑n
    this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
    ⊢ Eq (Int.fract (HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑(HMod.hMod m ↑n) ↑n)
  -/
  obtain ⟨m₀, rfl | rfl⟩ := m.eq_nat_or_neg
    /-
      case inr.intro.inl
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      ⊢ Eq (Int.fract (HDiv.hDiv ↑↑m₀ ↑n)) (HDiv.hDiv ↑(HMod.hMod ↑m₀ ↑n) ↑n)
    -/
  · exact this (ofNat_nonneg m₀)
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    n : Nat
    hn : LT.lt 0 ↑n
    this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
    m₀ : Nat
    ⊢ Eq (Int.fract (HDiv.hDiv ↑(Neg.neg ↑m₀) ↑n)) (HDiv.hDiv ↑(HMod.hMod (Neg.neg …
  -/
  let q := ⌈↑m₀ / (n : k)⌉
  /-
    case inr.intro.inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    n : Nat
    hn : LT.lt 0 ↑n
    this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
    m₀ : Nat
    q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
    ⊢ Eq (Int.fract (HDiv.hDiv ↑(Neg.neg ↑m₀) ↑n)) (HDiv.hDiv ↑(HMod.hMod (Neg.neg …
  -/
  let m₁ := q * ↑n - (↑m₀ : ℤ)
  have hm₁ : 0 ≤ m₁ := by
    simpa [m₁, ← @cast_le k, ← div_le_iff₀ hn] using FloorRing.gc_ceil_coe.le_u_l _
  calc
    fract ((Int.cast (-(m₀ : ℤ)) : k) / (n : k))
      -- Porting note: the `rw [cast_neg, cast_natCast]` was `push_cast`
      = fract (-(m₀ : k) / n) := by rw [cast_neg, cast_natCast]
    _ = fract ((m₁ : k) / n) := ?_
    _ = Int.cast (m₁ % (n : ℤ)) / Nat.cast n := this hm₁
    _ = Int.cast (-(↑m₀ : ℤ) % ↑n) / Nat.cast n := ?_

    /-
      case inr.intro.inr.calc_1
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
      m₁ : Int := HSub.hSub (HMul.hMul q ↑n) ↑m₀
      hm₁ : LE.le 0 m₁
      ⊢ Eq (Int.fract (HDiv.hDiv (Neg.neg ↑m₀) ↑n)) (Int.fract (HDiv.hDiv ↑m₁ ↑n))
    -/
  · rw [← fract_int_add q, ← mul_div_cancel_right₀ (q : k) hn.ne', ← add_div, ← sub_eq_add_neg]
    -- Porting note: the `simp` was `push_cast`
    /-
      case inr.intro.inr.calc_1
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
      m₁ : Int := HSub.hSub (HMul.hMul q ↑n) ↑m₀
      hm₁ : LE.le 0 m₁
      ⊢ Eq (Int.fract (HDiv.hDiv (HSub.hSub (HMul.hMul ↑q ↑n) ↑m₀) ↑n)) (Int.fract ( …
    -/
    simp [m₁]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.calc_2
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
      m₁ : Int := HSub.hSub (HMul.hMul q ↑n) ↑m₀
      hm₁ : LE.le 0 m₁
      ⊢ Eq (HDiv.hDiv ↑(HMod.hMod m₁ ↑n) ↑n) (HDiv.hDiv ↑(HMod.hMod (Neg.neg ↑m₀) ↑n …
    -/
  · congr 2
    /-
      case inr.intro.inr.calc_2.e_a.e_a
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
      m₁ : Int := HSub.hSub (HMul.hMul q ↑n) ↑m₀
      hm₁ : LE.le 0 m₁
      ⊢ Eq (HMod.hMod m₁ ↑n) (HMod.hMod (Neg.neg ↑m₀) ↑n)
    -/
    change (q * ↑n - (↑m₀ : ℤ)) % ↑n = _
    /-
      case inr.intro.inr.calc_2.e_a.e_a
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      n : Nat
      hn : LT.lt 0 ↑n
      this : ∀ {l : Int}, LE.le 0 l → Eq (Int.fract (HDiv.hDiv ↑l ↑n)) (HDiv.hDiv ↑( …
      m₀ : Nat
      q : Int := Int.ceil (HDiv.hDiv ↑m₀ ↑n)
      m₁ : Int := HSub.hSub (HMul.hMul q ↑n) ↑m₀
      hm₁ : LE.le 0 m₁
      ⊢ Eq (HMod.hMod (HSub.hSub (HMul.hMul q ↑n) ↑m₀) ↑n) (HMod.hMod (Neg.neg ↑m₀)  …
    -/
    rw [sub_eq_add_neg, add_comm (q * ↑n), add_mul_emod_self]
    /-
      🎉 no goals
    -/


theorem gc_ceil_coe : GaloisConnection ceil ((↑) : ℤ → α) :=
  FloorRing.gc_ceil_coe


theorem ceil_le : ⌈a⌉ ≤ z ↔ a ≤ z :=
  gc_ceil_coe a z


theorem floor_neg : ⌊-a⌋ = -⌈a⌉ :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    a : α
                                    z : Int
                                    ⊢ Iff (LE.le z (Int.floor (Neg.neg a))) (LE.le z (Neg.neg (Int.ceil a)))
                                  -/
  eq_of_forall_le_iff fun z => by rw [le_neg, ceil_le, le_floor, Int.cast_neg, le_neg]
                                  /-
                                    🎉 no goals
                                  -/


theorem ceil_neg : ⌈-a⌉ = -⌊a⌋ :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    a : α
                                    z : Int
                                    ⊢ Iff (LE.le (Int.ceil (Neg.neg a)) z) (LE.le (Neg.neg (Int.floor a)) z)
                                  -/
  eq_of_forall_ge_iff fun z => by rw [neg_le, ceil_le, le_floor, Int.cast_neg, neg_le]
                                  /-
                                    🎉 no goals
                                  -/


theorem lt_ceil : z < ⌈a⌉ ↔ (z : α) < a :=
  lt_iff_lt_of_le_iff_le ceil_le


@[simp]
                                                              /-
                                                                α : Type u_2
                                                                inst✝¹ : LinearOrderedRing α
                                                                inst✝ : FloorRing α
                                                                z : Int
                                                                a : α
                                                                ⊢ Iff (LE.le (HAdd.hAdd z 1) (Int.ceil a)) (LT.lt (↑z) a)
                                                              -/
theorem add_one_le_ceil_iff : z + 1 ≤ ⌈a⌉ ↔ (z : α) < a := by rw [← lt_ceil, add_one_le_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem one_le_ceil_iff : 1 ≤ ⌈a⌉ ↔ 0 < a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Iff (LE.le 1 (Int.ceil a)) (LT.lt 0 a)
  -/
  rw [← zero_add (1 : ℤ), add_one_le_ceil_iff, cast_zero]
  /-
    🎉 no goals
  -/


@[bound]
theorem ceil_le_floor_add_one (a : α) : ⌈a⌉ ≤ ⌊a⌋ + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ LE.le (Int.ceil a) (HAdd.hAdd (Int.floor a) 1)
  -/
  rw [ceil_le, Int.cast_add, Int.cast_one]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ LE.le a (HAdd.hAdd (↑(Int.floor a)) 1)
  -/
  exact (lt_floor_add_one a).le
  /-
    🎉 no goals
  -/


@[bound]
theorem le_ceil (a : α) : a ≤ ⌈a⌉ :=
  gc_ceil_coe.le_u_l a


                                              /-
                                                α : Type u_2
                                                inst✝¹ : LinearOrderedRing α
                                                inst✝ : FloorRing α
                                                z : Int
                                                a : α
                                                ⊢ Iff (LE.le z (Int.ceil a)) (LT.lt (HSub.hSub (↑z) 1) a)
                                              -/
lemma le_ceil_iff : z ≤ ⌈a⌉ ↔ z - 1 < a := by rw [← sub_one_lt_iff, lt_ceil]; norm_cast
                                                                              /-
                                                                                🎉 no goals
                                                                              -/

                                              /-
                                                α : Type u_2
                                                inst✝¹ : LinearOrderedRing α
                                                inst✝ : FloorRing α
                                                z : Int
                                                a : α
                                                ⊢ Iff (LT.lt (Int.ceil a) z) (LE.le a (HSub.hSub (↑z) 1))
                                              -/
lemma ceil_lt_iff : ⌈a⌉ < z ↔ a ≤ z - 1 := by rw [← le_sub_one_iff, ceil_le]; norm_cast
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem ceil_intCast (z : ℤ) : ⌈(z : α)⌉ = z :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    z a : Int
                                    ⊢ Iff (LE.le (Int.ceil ↑z) a) (LE.le z a)
                                  -/
  eq_of_forall_ge_iff fun a => by rw [ceil_le, Int.cast_le]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem ceil_natCast (n : ℕ) : ⌈(n : α)⌉ = n :=
                                  /-
                                    α : Type u_2
                                    inst✝¹ : LinearOrderedRing α
                                    inst✝ : FloorRing α
                                    n : Nat
                                    a : Int
                                    ⊢ Iff (LE.le (Int.ceil ↑n) a) (LE.le (↑n) a)
                                  -/
  eq_of_forall_ge_iff fun a => by rw [ceil_le, ← cast_natCast, cast_le]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem ceil_ofNat (n : ℕ) [n.AtLeastTwo] : ⌈(ofNat(n) : α)⌉ = ofNat(n) := ceil_natCast n


theorem ceil_mono : Monotone (ceil : α → ℤ) :=
  gc_ceil_coe.monotone_l


@[gcongr, bound] lemma ceil_le_ceil (hab : a ≤ b) : ⌈a⌉ ≤ ⌈b⌉ := ceil_mono hab


@[simp]
theorem ceil_add_int (a : α) (z : ℤ) : ⌈a + z⌉ = ⌈a⌉ + z := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    z : Int
    ⊢ Eq (Int.ceil (HAdd.hAdd a ↑z)) (HAdd.hAdd (Int.ceil a) z)
  -/
  rw [← neg_inj, neg_add', ← floor_neg, ← floor_neg, neg_add', floor_sub_int]
  /-
    🎉 no goals
  -/


@[simp]
                                                               /-
                                                                 α : Type u_2
                                                                 inst✝¹ : LinearOrderedRing α
                                                                 inst✝ : FloorRing α
                                                                 a : α
                                                                 n : Nat
                                                                 ⊢ Eq (Int.ceil (HAdd.hAdd a ↑n)) (HAdd.hAdd (Int.ceil a) ↑n)
                                                               -/
theorem ceil_add_nat (a : α) (n : ℕ) : ⌈a + n⌉ = ⌈a⌉ + n := by rw [← Int.cast_natCast, ceil_add_int]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem ceil_add_one (a : α) : ⌈a + 1⌉ = ⌈a⌉ + 1 := by
  -- Porting note: broken `convert ceil_add_int a (1 : ℤ)`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Int.ceil (HAdd.hAdd a 1)) (HAdd.hAdd (Int.ceil a) 1)
  -/
  rw [← ceil_add_int a (1 : ℤ), cast_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem ceil_add_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌈a + ofNat(n)⌉ = ⌈a⌉ + ofNat(n) :=
  ceil_add_nat a n


@[simp]
theorem ceil_sub_int (a : α) (z : ℤ) : ⌈a - z⌉ = ⌈a⌉ - z :=
               /-
                 α : Type u_2
                 inst✝¹ : LinearOrderedRing α
                 inst✝ : FloorRing α
                 a : α
                 z : Int
                 ⊢ Eq (Int.ceil (HSub.hSub a ↑z)) (Int.ceil (HAdd.hAdd a ↑(Neg.neg z)))
               -/
  Eq.trans (by rw [Int.cast_neg, sub_eq_add_neg]) (ceil_add_int _ _)
               /-
                 🎉 no goals
               -/


@[simp]
theorem ceil_sub_nat (a : α) (n : ℕ) : ⌈a - n⌉ = ⌈a⌉ - n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (Int.ceil (HSub.hSub a ↑n)) (HSub.hSub (Int.ceil a) ↑n)
  -/
  convert ceil_sub_int a n using 1
  /-
    case h.e'_2
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    n : Nat
    ⊢ Eq (Int.ceil (HSub.hSub a ↑n)) (Int.ceil (HSub.hSub a ↑↑n))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem ceil_sub_one (a : α) : ⌈a - 1⌉ = ⌈a⌉ - 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Int.ceil (HSub.hSub a 1)) (HSub.hSub (Int.ceil a) 1)
  -/
  rw [eq_sub_iff_add_eq, ← ceil_add_one, sub_add_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem ceil_sub_ofNat (a : α) (n : ℕ) [n.AtLeastTwo] :
    ⌈a - ofNat(n)⌉ = ⌈a⌉ - ofNat(n) :=
  ceil_sub_nat a n


@[bound]
theorem ceil_lt_add_one (a : α) : (⌈a⌉ : α) < a + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ LT.lt (↑(Int.ceil a)) (HAdd.hAdd a 1)
  -/
  rw [← lt_ceil, ← Int.cast_one, ceil_add_int]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ LT.lt (Int.ceil a) (HAdd.hAdd (Int.ceil a) 1)
  -/
  apply lt_add_one
  /-
    🎉 no goals
  -/


@[bound]
theorem ceil_add_le (a b : α) : ⌈a + b⌉ ≤ ⌈a⌉ + ⌈b⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (Int.ceil (HAdd.hAdd a b)) (HAdd.hAdd (Int.ceil a) (Int.ceil b))
  -/
  rw [ceil_le, Int.cast_add]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd ↑(Int.ceil a) ↑(Int.ceil b))
  -/
             /-
               🎉 no goals
             -/
  gcongr <;> apply le_ceil
             /-
               🎉 no goals
             -/


@[bound]
theorem ceil_add_ceil_le (a b : α) : ⌈a⌉ + ⌈b⌉ ≤ ⌈a + b⌉ + 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd (Int.ceil a) (Int.ceil b)) (HAdd.hAdd (Int.ceil (HAdd.hAdd  …
  -/
  rw [← le_sub_iff_add_le, ceil_le, Int.cast_sub, Int.cast_add, Int.cast_one, le_sub_comm]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (↑(Int.ceil b)) (HSub.hSub (HAdd.hAdd (↑(Int.ceil (HAdd.hAdd a b))) 1) …
  -/
  refine (ceil_lt_add_one _).le.trans ?_
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd b 1) (HSub.hSub (HAdd.hAdd (↑(Int.ceil (HAdd.hAdd a b))) 1) …
  -/
  rw [le_sub_iff_add_le', ← add_assoc, add_le_add_iff_right]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ LE.le (HAdd.hAdd a b) ↑(Int.ceil (HAdd.hAdd a b))
  -/
  exact le_ceil _
  /-
    🎉 no goals
  -/


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝¹ : LinearOrderedRing α
                                           inst✝ : FloorRing α
                                           a : α
                                           ⊢ Iff (LT.lt 0 (Int.ceil a)) (LT.lt 0 a)
                                         -/
theorem ceil_pos : 0 < ⌈a⌉ ↔ 0 < a := by rw [lt_ceil, cast_zero]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                        /-
                                          α : Type u_2
                                          inst✝¹ : LinearOrderedRing α
                                          inst✝ : FloorRing α
                                          ⊢ Eq (Int.ceil 0) 0
                                        -/
theorem ceil_zero : ⌈(0 : α)⌉ = 0 := by rw [← cast_zero, ceil_intCast]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
                                       /-
                                         α : Type u_2
                                         inst✝¹ : LinearOrderedRing α
                                         inst✝ : FloorRing α
                                         ⊢ Eq (Int.ceil 1) 1
                                       -/
theorem ceil_one : ⌈(1 : α)⌉ = 1 := by rw [← cast_one, ceil_intCast]
                                       /-
                                         🎉 no goals
                                       -/


@[bound]
theorem ceil_nonneg (ha : 0 ≤ a) : 0 ≤ ⌈a⌉ := mod_cast ha.trans (le_ceil a)


theorem ceil_eq_iff : ⌈a⌉ = z ↔ ↑z - 1 < a ∧ a ≤ z := by
  rw [← ceil_le, ← Int.cast_one, ← Int.cast_sub, ← lt_ceil, Int.sub_one_lt_iff, le_antisymm_iff,
    and_comm]


@[simp]
                                                              /-
                                                                α : Type u_2
                                                                inst✝¹ : LinearOrderedRing α
                                                                inst✝ : FloorRing α
                                                                a : α
                                                                ⊢ Iff (Eq (Int.ceil a) 0) (Membership.mem (Set.Ioc (-1) 0) a)
                                                              -/
theorem ceil_eq_zero_iff : ⌈a⌉ = 0 ↔ a ∈ Ioc (-1 : α) 0 := by simp [ceil_eq_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem ceil_eq_on_Ioc (z : ℤ) : ∀ a ∈ Set.Ioc (z - 1 : α) z, ⌈a⌉ = z := fun _ ⟨h₀, h₁⟩ =>
  ceil_eq_iff.mpr ⟨h₀, h₁⟩


theorem ceil_eq_on_Ioc' (z : ℤ) : ∀ a ∈ Set.Ioc (z - 1 : α) z, (⌈a⌉ : α) = z := fun a ha =>
  mod_cast ceil_eq_on_Ioc z a ha


@[bound]
theorem floor_le_ceil (a : α) : ⌊a⌋ ≤ ⌈a⌉ :=
  cast_le.1 <| (floor_le _).trans <| le_ceil _


@[bound]
theorem floor_lt_ceil_of_lt {a b : α} (h : a < b) : ⌊a⌋ < ⌈b⌉ :=
  cast_lt.1 <| (floor_le a).trans_lt <| h.trans_le <| le_ceil b

-- Porting note: in mathlib3 there was no need for the type annotation in `(m : α)`

@[simp]
theorem preimage_ceil_singleton (m : ℤ) : (ceil : α → ℤ) ⁻¹' {m} = Ioc ((m : α) - 1) m :=
  ext fun _ => ceil_eq_iff


theorem fract_eq_zero_or_add_one_sub_ceil (a : α) : fract a = 0 ∨ fract a = a + 1 - (⌈a⌉ : α) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Or (Eq (Int.fract a) 0) (Eq (Int.fract a) (HSub.hSub (HAdd.hAdd a 1) ↑(Int.c …
  -/
  rcases eq_or_ne (fract a) 0 with ha | ha
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      a : α
      ha : Eq (Int.fract a) 0
      ⊢ Or (Eq (Int.fract a) 0) (Eq (Int.fract a) (HSub.hSub (HAdd.hAdd a 1) ↑(Int.c …
    -/
  · exact Or.inl ha
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Or (Eq (Int.fract a) 0) (Eq (Int.fract a) (HSub.hSub (HAdd.hAdd a 1) ↑(Int.c …
  -/
  right
  suffices (⌈a⌉ : α) = ⌊a⌋ + 1 by
    rw [this, ← self_sub_fract]
    abel
  /-
    case inr.h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (↑(Int.ceil a)) (HAdd.hAdd (↑(Int.floor a)) 1)
  -/
  norm_cast
  /-
    case inr.h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (Int.ceil a) (HAdd.hAdd (Int.floor a) 1)
  -/
  rw [ceil_eq_iff]
  /-
    case inr.h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ And (LT.lt (HSub.hSub (↑(HAdd.hAdd (Int.floor a) 1)) 1) a) (LE.le a ↑(HAdd.h …
  -/
  refine ⟨?_, _root_.le_of_lt <| by simp⟩
  /-
    case inr.h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ LT.lt (HSub.hSub (↑(HAdd.hAdd (Int.floor a) 1)) 1) a
  -/
  rw [cast_add, cast_one, add_tsub_cancel_right, ← self_sub_fract a, sub_lt_self_iff]
  /-
    case inr.h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ LT.lt 0 (Int.fract a)
  -/
  exact ha.symm.lt_of_le (fract_nonneg a)
  /-
    🎉 no goals
  -/


theorem ceil_eq_add_one_sub_fract (ha : fract a ≠ 0) : (⌈a⌉ : α) = a + 1 - fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (↑(Int.ceil a)) (HSub.hSub (HAdd.hAdd a 1) (Int.fract a))
  -/
  rw [(or_iff_right ha).mp (fract_eq_zero_or_add_one_sub_ceil a)]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (↑(Int.ceil a)) (HSub.hSub (HAdd.hAdd a 1) (HSub.hSub (HAdd.hAdd a 1) ↑(I …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem ceil_sub_self_eq (ha : fract a ≠ 0) : (⌈a⌉ : α) - a = 1 - fract a := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (HSub.hSub (↑(Int.ceil a)) a) (HSub.hSub 1 (Int.fract a))
  -/
  rw [(or_iff_right ha).mp (fract_eq_zero_or_add_one_sub_ceil a)]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : Ne (Int.fract a) 0
    ⊢ Eq (HSub.hSub (↑(Int.ceil a)) a) (HSub.hSub 1 (HSub.hSub (HAdd.hAdd a 1) ↑(I …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma mul_lt_floor (hb₀ : 0 < b) (hb : b < 1) (hba : ⌈b / (1 - b)⌉ ≤ a) : b * a < ⌊a⌋ := by
  calc
    b * a < b * (⌊a⌋ + 1) := by gcongr; exacts [hb₀, lt_floor_add_one _]
    _ ≤ ⌊a⌋ := by
      rwa [_root_.mul_add_one, ← le_sub_iff_add_le', ← one_sub_mul, ← div_le_iff₀' (by linarith),
        ← ceil_le, le_floor]


lemma ceil_div_ceil_inv_sub_one (ha : 1 ≤ a) : ⌈⌈(a - 1)⁻¹⌉ / a⌉ = ⌈(a - 1)⁻¹⌉ := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha : LE.le 1 a
    ⊢ Eq (Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub a 1)))) a)) (Int.cei …
  -/
  obtain rfl | ha := ha.eq_or_lt
    /-
      case inl
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      ha : LE.le 1 1
      ⊢ Eq (Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub 1 1)))) 1)) (Int.cei …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha✝ : LE.le 1 a
    ha : LT.lt 1 a
    ⊢ Eq (Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub a 1)))) a)) (Int.cei …
  -/
  have : 0 < a - 1 := by linarith
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha✝ : LE.le 1 a
    ha : LT.lt 1 a
    this : LT.lt 0 (HSub.hSub a 1)
    ⊢ Eq (Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub a 1)))) a)) (Int.cei …
  -/
  have : 0 < ⌈(a - 1)⁻¹⌉ := ceil_pos.2 <| by positivity
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha✝ : LE.le 1 a
    ha : LT.lt 1 a
    this✝ : LT.lt 0 (HSub.hSub a 1)
    this : LT.lt 0 (Int.ceil (Inv.inv (HSub.hSub a 1)))
    ⊢ Eq (Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub a 1)))) a)) (Int.cei …
  -/
  refine le_antisymm (ceil_le.2 <| div_le_self (by positivity) ha.le) <| ?_
  rw [le_ceil_iff, sub_lt_comm, div_eq_mul_inv, ← mul_one_sub,
    ← lt_div_iff₀ (sub_pos.2 <| inv_lt_one_of_one_lt₀ ha)]
  /-
    case inr
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha✝ : LE.le 1 a
    ha : LT.lt 1 a
    this✝ : LT.lt 0 (HSub.hSub a 1)
    this : LT.lt 0 (Int.ceil (Inv.inv (HSub.hSub a 1)))
    ⊢ LT.lt (↑(Int.ceil (Inv.inv (HSub.hSub a 1)))) (HDiv.hDiv 1 (HSub.hSub 1 (Inv …
  -/
  convert ceil_lt_add_one _ using 1
  /-
    case h.e'_4
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha✝ : LE.le 1 a
    ha : LT.lt 1 a
    this✝ : LT.lt 0 (HSub.hSub a 1)
    this : LT.lt 0 (Int.ceil (Inv.inv (HSub.hSub a 1)))
    ⊢ Eq (HDiv.hDiv 1 (HSub.hSub 1 (Inv.inv a))) (HAdd.hAdd (Inv.inv (HSub.hSub a  …
  -/
  field_simp
  /-
    🎉 no goals
  -/


lemma ceil_lt_mul (hb : 1 < b) (hba : ⌈(b - 1)⁻¹⌉ / b < a) : ⌈a⌉ < b * a := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a b : k
    hb : LT.lt 1 b
    hba : LT.lt (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) a
    ⊢ LT.lt (↑(Int.ceil a)) (HMul.hMul b a)
  -/
  obtain hab | hba := le_total a (b - 1)⁻¹
  · calc
      ⌈a⌉ ≤ (⌈(b - 1)⁻¹⌉ : k) := by gcongr
      _ < b * a := by rwa [← div_lt_iff₀']; positivity
    /-
      case inr
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      a b : k
      hb : LT.lt 1 b
      hba✝ : LT.lt (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      hba : LE.le (Inv.inv (HSub.hSub b 1)) a
      ⊢ LT.lt (↑(Int.ceil a)) (HMul.hMul b a)
    -/
  · rw [← sub_pos] at hb
    calc
      ⌈a⌉ < a + 1 := ceil_lt_add_one _
      _ = a + (b - 1) * (b - 1)⁻¹ := by rw [mul_inv_cancel₀]; positivity
      _ ≤ a + (b - 1) * a := by gcongr; positivity
      _ = b * a := by rw [sub_one_mul, add_sub_cancel]


lemma ceil_le_mul (hb : 1 < b) (hba : ⌈(b - 1)⁻¹⌉ / b ≤ a) : ⌈a⌉ ≤ b * a := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a b : k
    hb : LT.lt 1 b
    hba : LE.le (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) a
    ⊢ LE.le (↑(Int.ceil a)) (HMul.hMul b a)
  -/
  obtain rfl | hba := hba.eq_or_lt
    /-
      case inl
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      b : k
      hb : LT.lt 1 b
      hba : LE.le (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) (HDiv.hDiv ( …
      ⊢ LE.le (↑(Int.ceil (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b))) (H …
    -/
  · rw [ceil_div_ceil_inv_sub_one hb.le, mul_div_cancel₀]
    /-
      case inl.hb
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      b : k
      hb : LT.lt 1 b
      hba : LE.le (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) (HDiv.hDiv ( …
      ⊢ Ne b 0
    -/
    positivity
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_4
      inst✝¹ : LinearOrderedField k
      inst✝ : FloorRing k
      a b : k
      hb : LT.lt 1 b
      hba✝ : LE.le (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      hba : LT.lt (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub b 1)))) b) a
      ⊢ LE.le (↑(Int.ceil a)) (HMul.hMul b a)
    -/
  · exact (ceil_lt_mul hb hba).le
    /-
      🎉 no goals
    -/


lemma div_two_lt_floor (ha : 1 ≤ a) : a / 2 < ⌊a⌋ := by
  /-
    k : Type u_4
    inst✝¹ : LinearOrderedField k
    inst✝ : FloorRing k
    a : k
    ha : LE.le 1 a
    ⊢ LT.lt (HDiv.hDiv a 2) ↑(Int.floor a)
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  rw [div_eq_inv_mul]; refine mul_lt_floor ?_ ?_ ?_ <;> norm_num; assumption
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma ceil_lt_two_mul (ha : 2⁻¹ < a) : ⌈a⌉ < 2 * a :=
                             /-
                               k : Type u_4
                               inst✝¹ : LinearOrderedField k
                               inst✝ : FloorRing k
                               a : k
                               ha : LT.lt (Inv.inv 2) a
                               ⊢ LT.lt (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub 2 1)))) 2) a
                             -/
  ceil_lt_mul one_lt_two (by norm_num at ha ⊢; exact ha)
                                               /-
                                                 🎉 no goals
                                               -/


lemma ceil_le_two_mul (ha : 2⁻¹ ≤ a) : ⌈a⌉ ≤ 2 * a :=
                             /-
                               k : Type u_4
                               inst✝¹ : LinearOrderedField k
                               inst✝ : FloorRing k
                               a : k
                               ha : LE.le (Inv.inv 2) a
                               ⊢ LE.le (HDiv.hDiv (↑(Int.ceil (Inv.inv (HSub.hSub 2 1)))) 2) a
                             -/
  ceil_le_mul one_lt_two (by norm_num at ha ⊢; exact ha)
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem preimage_Ioo {a b : α} : ((↑) : ℤ → α) ⁻¹' Set.Ioo a b = Set.Ioo ⌊a⌋ ⌈b⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ Eq (Set.preimage Int.cast (Set.Ioo a b)) (Set.Ioo (Int.floor a) (Int.ceil b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Ioo a b)) x✝) (Membership.me …
  -/
  simp [floor_lt, lt_ceil]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ico {a b : α} : ((↑) : ℤ → α) ⁻¹' Set.Ico a b = Set.Ico ⌈a⌉ ⌈b⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ Eq (Set.preimage Int.cast (Set.Ico a b)) (Set.Ico (Int.ceil a) (Int.ceil b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Ico a b)) x✝) (Membership.me …
  -/
  simp [ceil_le, lt_ceil]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ioc {a b : α} : ((↑) : ℤ → α) ⁻¹' Set.Ioc a b = Set.Ioc ⌊a⌋ ⌊b⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ Eq (Set.preimage Int.cast (Set.Ioc a b)) (Set.Ioc (Int.floor a) (Int.floor b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Ioc a b)) x✝) (Membership.me …
  -/
  simp [floor_lt, le_floor]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Icc {a b : α} : ((↑) : ℤ → α) ⁻¹' Set.Icc a b = Set.Icc ⌈a⌉ ⌊b⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    ⊢ Eq (Set.preimage Int.cast (Set.Icc a b)) (Set.Icc (Int.ceil a) (Int.floor b))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a b : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Icc a b)) x✝) (Membership.me …
  -/
  simp [ceil_le, le_floor]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ioi : ((↑) : ℤ → α) ⁻¹' Set.Ioi a = Set.Ioi ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Set.preimage Int.cast (Set.Ioi a)) (Set.Ioi (Int.floor a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Ioi a)) x✝) (Membership.mem  …
  -/
  simp [floor_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Ici : ((↑) : ℤ → α) ⁻¹' Set.Ici a = Set.Ici ⌈a⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Set.preimage Int.cast (Set.Ici a)) (Set.Ici (Int.ceil a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Ici a)) x✝) (Membership.mem  …
  -/
  simp [ceil_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Iio : ((↑) : ℤ → α) ⁻¹' Set.Iio a = Set.Iio ⌈a⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Set.preimage Int.cast (Set.Iio a)) (Set.Iio (Int.ceil a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Iio a)) x✝) (Membership.mem  …
  -/
  simp [lt_ceil]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_Iic : ((↑) : ℤ → α) ⁻¹' Set.Iic a = Set.Iic ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (Set.preimage Int.cast (Set.Iic a)) (Set.Iic (Int.floor a))
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    x✝ : Int
    ⊢ Iff (Membership.mem (Set.preimage Int.cast (Set.Iic a)) x✝) (Membership.mem  …
  -/
  simp [le_floor]
  /-
    🎉 no goals
  -/


/-- `round` rounds a number to the nearest integer. `round (1 / 2) = 1` -/
def round (x : α) : ℤ :=
  if 2 * fract x < 1 then ⌊x⌋ else ⌈x⌉


@[simp]
                                             /-
                                               α : Type u_2
                                               inst✝¹ : LinearOrderedRing α
                                               inst✝ : FloorRing α
                                               ⊢ Eq (round 0) 0
                                             -/
theorem round_zero : round (0 : α) = 0 := by simp [round]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                            /-
                                              α : Type u_2
                                              inst✝¹ : LinearOrderedRing α
                                              inst✝ : FloorRing α
                                              ⊢ Eq (round 1) 1
                                            -/
theorem round_one : round (1 : α) = 1 := by simp [round]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : LinearOrderedRing α
                                                          inst✝ : FloorRing α
                                                          n : Nat
                                                          ⊢ Eq (round ↑n) ↑n
                                                        -/
theorem round_natCast (n : ℕ) : round (n : α) = n := by simp [round]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem round_ofNat (n : ℕ) [n.AtLeastTwo] : round (ofNat(n) : α) = ofNat(n) :=
  round_natCast n


@[simp]
                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : LinearOrderedRing α
                                                          inst✝ : FloorRing α
                                                          n : Int
                                                          ⊢ Eq (round ↑n) n
                                                        -/
theorem round_intCast (n : ℤ) : round (n : α) = n := by simp [round]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem round_add_int (x : α) (y : ℤ) : round (x + y) = round x + y := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Int
    ⊢ Eq (round (HAdd.hAdd x ↑y)) (HAdd.hAdd (round x) y)
  -/
  rw [round, round, Int.fract_add_int, Int.floor_add_int, Int.ceil_add_int, ← apply_ite₂, ite_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_add_one (a : α) : round (a + 1) = round a + 1 := by
  -- Porting note: broken `convert round_add_int a 1`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (round (HAdd.hAdd a 1)) (HAdd.hAdd (round a) 1)
  -/
  rw [← round_add_int a 1, cast_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_sub_int (x : α) (y : ℤ) : round (x - y) = round x - y := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Int
    ⊢ Eq (round (HSub.hSub x ↑y)) (HSub.hSub (round x) y)
  -/
  rw [sub_eq_add_neg]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Int
    ⊢ Eq (round (HAdd.hAdd x (Neg.neg ↑y))) (HSub.hSub (round x) y)
  -/
  norm_cast
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Int
    ⊢ Eq (round (HAdd.hAdd x ↑(Neg.neg y))) (HSub.hSub (round x) y)
  -/
  rw [round_add_int, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_sub_one (a : α) : round (a - 1) = round a - 1 := by
  -- Porting note: broken `convert round_sub_int a 1`
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ⊢ Eq (round (HSub.hSub a 1)) (HSub.hSub (round a) 1)
  -/
  rw [← round_sub_int a 1, cast_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_add_nat (x : α) (y : ℕ) : round (x + y) = round x + y :=
  mod_cast round_add_int x y


@[simp]
theorem round_add_ofNat (x : α) (n : ℕ) [n.AtLeastTwo] :
    round (x + ofNat(n)) = round x + ofNat(n) :=
  round_add_nat x n


@[simp]
theorem round_sub_nat (x : α) (y : ℕ) : round (x - y) = round x - y :=
  mod_cast round_sub_int x y


@[simp]
theorem round_sub_ofNat (x : α) (n : ℕ) [n.AtLeastTwo] :
    round (x - ofNat(n)) = round x - ofNat(n) :=
  round_sub_nat x n


@[simp]
theorem round_int_add (x : α) (y : ℤ) : round ((y : α) + x) = y + round x := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Int
    ⊢ Eq (round (HAdd.hAdd (↑y) x)) (HAdd.hAdd y (round x))
  -/
  rw [add_comm, round_add_int, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_nat_add (x : α) (y : ℕ) : round ((y : α) + x) = y + round x := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    y : Nat
    ⊢ Eq (round (HAdd.hAdd (↑y) x)) (HAdd.hAdd (↑y) (round x))
  -/
  rw [add_comm, round_add_nat, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_ofNat_add (n : ℕ) [n.AtLeastTwo] (x : α) :
    round (ofNat(n) + x) = ofNat(n) + round x :=
  round_nat_add x n


theorem abs_sub_round_eq_min (x : α) : |x - round x| = min (fract x) (1 - fract x) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    ⊢ Eq (abs (HSub.hSub x ↑(round x))) (Min.min (Int.fract x) (HSub.hSub 1 (Int.f …
  -/
  simp_rw [round, min_def_lt, two_mul, ← lt_tsub_iff_left]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    ⊢ Eq (abs (HSub.hSub x ↑(ite (LT.lt (Int.fract x) (HSub.hSub 1 (Int.fract x))) …
  -/
  cases' lt_or_ge (fract x) (1 - fract x) with hx hx
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      hx : LT.lt (Int.fract x) (HSub.hSub 1 (Int.fract x))
      ⊢ Eq (abs (HSub.hSub x ↑(ite (LT.lt (Int.fract x) (HSub.hSub 1 (Int.fract x))) …
    -/
  · rw [if_pos hx, if_pos hx, self_sub_floor, abs_fract]
    /-
      🎉 no goals
    -/
  · have : 0 < fract x := by
      replace hx : 0 < fract x + fract x := lt_of_lt_of_le zero_lt_one (tsub_le_iff_left.mp hx)
      simpa only [← two_mul, mul_pos_iff_of_pos_left, zero_lt_two] using hx
    rw [if_neg (not_lt.mpr hx), if_neg (not_lt.mpr hx), abs_sub_comm, ceil_sub_self_eq this.ne.symm,
      abs_one_sub_fract]


theorem round_le (x : α) (z : ℤ) : |x - round x| ≤ |x - z| := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    z : Int
    ⊢ LE.le (abs (HSub.hSub x ↑(round x))) (abs (HSub.hSub x ↑z))
  -/
  rw [abs_sub_round_eq_min, min_le_iff]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    x : α
    z : Int
    ⊢ Or (LE.le (Int.fract x) (abs (HSub.hSub x ↑z))) (LE.le (HSub.hSub 1 (Int.fra …
  -/
  rcases le_or_lt (z : α) x with (hx | hx) <;> [left; right]
    /-
      case inl.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LE.le (↑z) x
      ⊢ LE.le (Int.fract x) (abs (HSub.hSub x ↑z))
    -/
  · conv_rhs => rw [abs_eq_self.mpr (sub_nonneg.mpr hx), ← fract_add_floor x, add_sub_assoc]
    /-
      case inl.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LE.le (↑z) x
      ⊢ LE.le (Int.fract x) (HAdd.hAdd (Int.fract x) (HSub.hSub ↑(Int.floor x) ↑z))
    -/
    simpa only [le_add_iff_nonneg_right, sub_nonneg, cast_le] using le_floor.mpr hx
    /-
      🎉 no goals
    -/
    /-
      case inr.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LT.lt x ↑z
      ⊢ LE.le (HSub.hSub 1 (Int.fract x)) (abs (HSub.hSub x ↑z))
    -/
  · rw [abs_eq_neg_self.mpr (sub_neg.mpr hx).le]
    /-
      case inr.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LT.lt x ↑z
      ⊢ LE.le (HSub.hSub 1 (Int.fract x)) (Neg.neg (HSub.hSub x ↑z))
    -/
    conv_rhs => rw [← fract_add_floor x]
    rw [add_sub_assoc, add_comm, neg_add, neg_sub, le_add_neg_iff_add_le, sub_add_cancel,
      le_sub_comm]
    /-
      case inr.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LT.lt x ↑z
      ⊢ LE.le (↑(Int.floor x)) (HSub.hSub (↑z) 1)
    -/
    norm_cast
    /-
      case inr.h
      α : Type u_2
      inst✝¹ : LinearOrderedRing α
      inst✝ : FloorRing α
      x : α
      z : Int
      hx : LT.lt x ↑z
      ⊢ LE.le (Int.floor x) (HSub.hSub z 1)
    -/
    exact floor_le_sub_one_iff.mpr hx
    /-
      🎉 no goals
    -/


theorem round_eq (x : α) : round x = ⌊x + 1 / 2⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ Eq (round x) (Int.floor (HAdd.hAdd x (1 / 2)))
  -/
  simp_rw [round, (by simp only [lt_div_iff₀', two_pos] : 2 * fract x < 1 ↔ fract x < 1 / 2)]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ Eq (ite (LT.lt (Int.fract x) (1 / 2)) (Int.floor x) (Int.ceil x)) (Int.floor …
  -/
  cases' lt_or_le (fract x) (1 / 2) with hx hx
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorRing α
      x : α
      hx : LT.lt (Int.fract x) (1 / 2)
      ⊢ Eq (ite (LT.lt (Int.fract x) (1 / 2)) (Int.floor x) (Int.ceil x)) (Int.floor …
    -/
  · conv_rhs => rw [← fract_add_floor x, add_assoc, add_left_comm, floor_int_add]
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorRing α
      x : α
      hx : LT.lt (Int.fract x) (1 / 2)
      ⊢ Eq (ite (LT.lt (Int.fract x) (1 / 2)) (Int.floor x) (Int.ceil x)) (HAdd.hAdd …
    -/
    rw [if_pos hx, self_eq_add_right, floor_eq_iff, cast_zero, zero_add]
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorRing α
      x : α
      hx : LT.lt (Int.fract x) (1 / 2)
      ⊢ And (LE.le 0 (HAdd.hAdd (Int.fract x) (1 / 2))) (LT.lt (HAdd.hAdd (Int.fract …
    -/
    constructor
      /-
        case inl.left
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorRing α
        x : α
        hx : LT.lt (Int.fract x) (1 / 2)
        ⊢ LE.le 0 (HAdd.hAdd (Int.fract x) (1 / 2))
      -/
    · linarith [fract_nonneg x]
      /-
        🎉 no goals
      -/
      /-
        case inl.right
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorRing α
        x : α
        hx : LT.lt (Int.fract x) (1 / 2)
        ⊢ LT.lt (HAdd.hAdd (Int.fract x) (1 / 2)) 1
      -/
    · linarith
      /-
        🎉 no goals
      -/
  · have : ⌊fract x + 1 / 2⌋ = 1 := by
      rw [floor_eq_iff]
      constructor
      · norm_num
        linarith
      · norm_num
        linarith [fract_lt_one x]
    rw [if_neg (not_lt.mpr hx), ← fract_add_floor x, add_assoc, add_left_comm, floor_int_add,
      ceil_add_int, add_comm _ ⌊x⌋, add_right_inj, ceil_eq_iff, this, cast_one, sub_self]
    /-
      case inr
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorRing α
      x : α
      hx : LE.le (1 / 2) (Int.fract x)
      this : Eq (Int.floor (HAdd.hAdd (Int.fract x) (1 / 2))) 1
      ⊢ And (LT.lt 0 (Int.fract x)) (LE.le (Int.fract x) 1)
    -/
    constructor
      /-
        case inr.left
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorRing α
        x : α
        hx : LE.le (1 / 2) (Int.fract x)
        this : Eq (Int.floor (HAdd.hAdd (Int.fract x) (1 / 2))) 1
        ⊢ LT.lt 0 (Int.fract x)
      -/
    · linarith
      /-
        🎉 no goals
      -/
      /-
        case inr.right
        α : Type u_2
        inst✝¹ : LinearOrderedField α
        inst✝ : FloorRing α
        x : α
        hx : LE.le (1 / 2) (Int.fract x)
        this : Eq (Int.floor (HAdd.hAdd (Int.fract x) (1 / 2))) 1
        ⊢ LE.le (Int.fract x) 1
      -/
    · linarith [fract_lt_one x]
      /-
        🎉 no goals
      -/


@[simp]
theorem round_two_inv : round (2⁻¹ : α) = 1 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    ⊢ Eq (round (Inv.inv 2)) 1
  -/
  simp only [round_eq, ← one_div, add_halves, floor_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_neg_two_inv : round (-2⁻¹ : α) = 0 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    ⊢ Eq (round (Neg.neg (Inv.inv 2))) 0
  -/
  simp only [round_eq, ← one_div, neg_add_cancel, floor_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem round_eq_zero_iff {x : α} : round x = 0 ↔ x ∈ Ico (-(1 / 2)) ((1 : α) / 2) := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ Iff (Eq (round x) 0) (Membership.mem (Set.Ico (Neg.neg (1 / 2)) (1 / 2)) x)
  -/
  rw [round_eq, floor_eq_zero_iff, add_mem_Ico_iff_left]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ Iff (Membership.mem (Set.Ico (HSub.hSub 0 (1 / 2)) (HSub.hSub 1 (1 / 2))) x) …
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem abs_sub_round (x : α) : |x - round x| ≤ 1 / 2 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ LE.le (abs (HSub.hSub x ↑(round x))) (1 / 2)
  -/
  rw [round_eq, abs_sub_le_iff]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ And (LE.le (HSub.hSub x ↑(Int.floor (HAdd.hAdd x (1 / 2)))) (1 / 2)) (LE.le  …
  -/
  have := floor_le (x + 1 / 2)
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    this : LE.le (↑(Int.floor (HAdd.hAdd x (1 / 2)))) (HAdd.hAdd x (1 / 2))
    ⊢ And (LE.le (HSub.hSub x ↑(Int.floor (HAdd.hAdd x (1 / 2)))) (1 / 2)) (LE.le  …
  -/
  have := lt_floor_add_one (x + 1 / 2)
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    this✝ : LE.le (↑(Int.floor (HAdd.hAdd x (1 / 2)))) (HAdd.hAdd x (1 / 2))
    this : LT.lt (HAdd.hAdd x (1 / 2)) (HAdd.hAdd (↑(Int.floor (HAdd.hAdd x (1 / 2 …
    ⊢ And (LE.le (HSub.hSub x ↑(Int.floor (HAdd.hAdd x (1 / 2)))) (1 / 2)) (LE.le  …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> linarith
                  /-
                    🎉 no goals
                  -/


theorem abs_sub_round_div_natCast_eq {m n : ℕ} :
    |(m : α) / n - round ((m : α) / n)| = ↑(min (m % n) (n - m % n)) / n := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    m n : Nat
    ⊢ Eq (abs (HSub.hSub (HDiv.hDiv ↑m ↑n) ↑(round (HDiv.hDiv ↑m ↑n)))) (HDiv.hDiv …
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      α : Type u_2
      inst✝¹ : LinearOrderedField α
      inst✝ : FloorRing α
      m : Nat
      ⊢ Eq (abs (HSub.hSub (HDiv.hDiv ↑m ↑0) ↑(round (HDiv.hDiv ↑m ↑0)))) (HDiv.hDiv …
    -/
  · simp
    /-
      🎉 no goals
    -/
  have hn' : 0 < (n : α) := by
    norm_cast
  rw [abs_sub_round_eq_min, Nat.cast_min, ← min_div_div_right hn'.le,
    fract_div_natCast_eq_div_natCast_mod, Nat.cast_sub (m.mod_lt hn).le, sub_div, div_self hn'.ne']


@[bound]
theorem sub_half_lt_round (x : α) : x - 1 / 2 < round x := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ LT.lt (HSub.hSub x (1 / 2)) ↑(round x)
  -/
  rw [round_eq x, show x - 1 / 2 = x + 1 / 2 - 1 by linarith]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ LT.lt (HSub.hSub (HAdd.hAdd x (1 / 2)) 1) ↑(Int.floor (HAdd.hAdd x (1 / 2)))
  -/
  exact Int.sub_one_lt_floor (x + 1 / 2)
  /-
    🎉 no goals
  -/


@[bound]
theorem round_le_add_half (x : α) : round x ≤ x + 1 / 2 := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ LE.le (↑(round x)) (HAdd.hAdd x (1 / 2))
  -/
  rw [round_eq x]
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : α
    ⊢ LE.le (↑(Int.floor (HAdd.hAdd x (1 / 2)))) (HAdd.hAdd x (1 / 2))
  -/
  exact Int.floor_le (x + 1 / 2)
  /-
    🎉 no goals
  -/


theorem floor_congr (h : ∀ n : ℕ, (n : α) ≤ a ↔ (n : β) ≤ b) : ⌊a⌋₊ = ⌊b⌋₊ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrderedSemiring α
    inst✝² : LinearOrderedSemiring β
    inst✝¹ : FloorSemiring α
    inst✝ : FloorSemiring β
    a : α
    b : β
    h : ∀ (n : Nat), Iff (LE.le (↑n) a) (LE.le (↑n) b)
    ⊢ Eq (Nat.floor a) (Nat.floor b)
  -/
  have h₀ : 0 ≤ a ↔ 0 ≤ b := by simpa only [cast_zero] using h 0
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrderedSemiring α
    inst✝² : LinearOrderedSemiring β
    inst✝¹ : FloorSemiring α
    inst✝ : FloorSemiring β
    a : α
    b : β
    h : ∀ (n : Nat), Iff (LE.le (↑n) a) (LE.le (↑n) b)
    h₀ : Iff (LE.le 0 a) (LE.le 0 b)
    ⊢ Eq (Nat.floor a) (Nat.floor b)
  -/
  obtain ha | ha := lt_or_le a 0
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝³ : LinearOrderedSemiring α
      inst✝² : LinearOrderedSemiring β
      inst✝¹ : FloorSemiring α
      inst✝ : FloorSemiring β
      a : α
      b : β
      h : ∀ (n : Nat), Iff (LE.le (↑n) a) (LE.le (↑n) b)
      h₀ : Iff (LE.le 0 a) (LE.le 0 b)
      ha : LT.lt a 0
      ⊢ Eq (Nat.floor a) (Nat.floor b)
    -/
  · rw [floor_of_nonpos ha.le, floor_of_nonpos (le_of_not_le <| h₀.not.mp ha.not_le)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrderedSemiring α
    inst✝² : LinearOrderedSemiring β
    inst✝¹ : FloorSemiring α
    inst✝ : FloorSemiring β
    a : α
    b : β
    h : ∀ (n : Nat), Iff (LE.le (↑n) a) (LE.le (↑n) b)
    h₀ : Iff (LE.le 0 a) (LE.le 0 b)
    ha : LE.le 0 a
    ⊢ Eq (Nat.floor a) (Nat.floor b)
  -/
  exact (le_floor <| (h _).1 <| floor_le ha).antisymm (le_floor <| (h _).2 <| floor_le <| h₀.1 ha)
  /-
    🎉 no goals
  -/


theorem ceil_congr (h : ∀ n : ℕ, a ≤ n ↔ b ≤ n) : ⌈a⌉₊ = ⌈b⌉₊ :=
  (ceil_le.2 <| (h _).2 <| le_ceil _).antisymm <| ceil_le.2 <| (h _).1 <| le_ceil _


theorem map_floor (f : F) (hf : StrictMono f) (a : α) : ⌊f a⌋₊ = ⌊a⌋₊ :=
                          /-
                            F : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝⁵ : LinearOrderedSemiring α
                            inst✝⁴ : LinearOrderedSemiring β
                            inst✝³ : FloorSemiring α
                            inst✝² : FloorSemiring β
                            inst✝¹ : FunLike F α β
                            inst✝ : RingHomClass F α β
                            f : F
                            hf : StrictMono ⇑f
                            a : α
                            n : Nat
                            ⊢ Iff (LE.le (↑n) (f a)) (LE.le (↑n) a)
                          -/
  floor_congr fun n => by rw [← map_natCast f, hf.le_iff_le]
                          /-
                            🎉 no goals
                          -/


theorem map_ceil (f : F) (hf : StrictMono f) (a : α) : ⌈f a⌉₊ = ⌈a⌉₊ :=
                         /-
                           F : Type u_1
                           α : Type u_2
                           β : Type u_3
                           inst✝⁵ : LinearOrderedSemiring α
                           inst✝⁴ : LinearOrderedSemiring β
                           inst✝³ : FloorSemiring α
                           inst✝² : FloorSemiring β
                           inst✝¹ : FunLike F α β
                           inst✝ : RingHomClass F α β
                           f : F
                           hf : StrictMono ⇑f
                           a : α
                           n : Nat
                           ⊢ Iff (LE.le (f a) ↑n) (LE.le a ↑n)
                         -/
  ceil_congr fun n => by rw [← map_natCast f, hf.le_iff_le]
                         /-
                           🎉 no goals
                         -/


theorem floor_congr (h : ∀ n : ℤ, (n : α) ≤ a ↔ (n : β) ≤ b) : ⌊a⌋ = ⌊b⌋ :=
  (le_floor.2 <| (h _).1 <| floor_le _).antisymm <| le_floor.2 <| (h _).2 <| floor_le _


theorem ceil_congr (h : ∀ n : ℤ, a ≤ n ↔ b ≤ n) : ⌈a⌉ = ⌈b⌉ :=
  (ceil_le.2 <| (h _).2 <| le_ceil _).antisymm <| ceil_le.2 <| (h _).1 <| le_ceil _


theorem map_floor (f : F) (hf : StrictMono f) (a : α) : ⌊f a⌋ = ⌊a⌋ :=
                          /-
                            F : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝⁵ : LinearOrderedRing α
                            inst✝⁴ : LinearOrderedRing β
                            inst✝³ : FloorRing α
                            inst✝² : FloorRing β
                            inst✝¹ : FunLike F α β
                            inst✝ : RingHomClass F α β
                            f : F
                            hf : StrictMono ⇑f
                            a : α
                            n : Int
                            ⊢ Iff (LE.le (↑n) (f a)) (LE.le (↑n) a)
                          -/
  floor_congr fun n => by rw [← map_intCast f, hf.le_iff_le]
                          /-
                            🎉 no goals
                          -/


theorem map_ceil (f : F) (hf : StrictMono f) (a : α) : ⌈f a⌉ = ⌈a⌉ :=
                         /-
                           F : Type u_1
                           α : Type u_2
                           β : Type u_3
                           inst✝⁵ : LinearOrderedRing α
                           inst✝⁴ : LinearOrderedRing β
                           inst✝³ : FloorRing α
                           inst✝² : FloorRing β
                           inst✝¹ : FunLike F α β
                           inst✝ : RingHomClass F α β
                           f : F
                           hf : StrictMono ⇑f
                           a : α
                           n : Int
                           ⊢ Iff (LE.le (f a) ↑n) (LE.le a ↑n)
                         -/
  ceil_congr fun n => by rw [← map_intCast f, hf.le_iff_le]
                         /-
                           🎉 no goals
                         -/


theorem map_fract (f : F) (hf : StrictMono f) (a : α) : fract (f a) = f (fract a) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrderedRing α
    inst✝⁴ : LinearOrderedRing β
    inst✝³ : FloorRing α
    inst✝² : FloorRing β
    inst✝¹ : FunLike F α β
    inst✝ : RingHomClass F α β
    f : F
    hf : StrictMono ⇑f
    a : α
    ⊢ Eq (Int.fract (f a)) (f (Int.fract a))
  -/
  simp_rw [fract, map_sub, map_intCast, map_floor _ hf]
  /-
    🎉 no goals
  -/


theorem map_round (f : F) (hf : StrictMono f) (a : α) : round (f a) = round a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrderedField α
    inst✝⁴ : LinearOrderedField β
    inst✝³ : FloorRing α
    inst✝² : FloorRing β
    inst✝¹ : FunLike F α β
    inst✝ : RingHomClass F α β
    f : F
    hf : StrictMono ⇑f
    a : α
    ⊢ Eq (round (f a)) (round a)
  -/
  have H : f 2 = 2 := map_natCast f 2
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrderedField α
    inst✝⁴ : LinearOrderedField β
    inst✝³ : FloorRing α
    inst✝² : FloorRing β
    inst✝¹ : FunLike F α β
    inst✝ : RingHomClass F α β
    f : F
    hf : StrictMono ⇑f
    a : α
    H : Eq (f 2) 2
    ⊢ Eq (round (f a)) (round a)
  -/
  simp_rw [round_eq, ← map_floor _ hf, map_add, one_div, map_inv₀, H]
  /-
    🎉 no goals
  -/
  -- Porting note: was
  -- simp_rw [round_eq, ← map_floor _ hf, map_add, one_div, map_inv₀, map_bit0, map_one]
  -- Would have thought that `map_natCast` would replace `map_bit0, map_one` but seems not


instance (priority := 100) FloorRing.toFloorSemiring : FloorSemiring α where
  floor a := ⌊a⌋.toNat
  ceil a := ⌈a⌉.toNat
  floor_of_neg {_} ha := Int.toNat_of_nonpos (Int.floor_nonpos ha.le)
                          /-
                            F : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝¹ : LinearOrderedRing α
                            inst✝ : FloorRing α
                            a : α
                            n : Nat
                            ha : LE.le 0 a
                            ⊢ Iff (LE.le n ((fun a => (Int.floor a).toNat) a)) (LE.le (↑n) a)
                          -/
  gc_floor {a n} ha := by rw [Int.le_toNat (Int.floor_nonneg.2 ha), Int.le_floor, Int.cast_natCast]
                          /-
                            🎉 no goals
                          -/
                    /-
                      F : Type u_1
                      α : Type u_2
                      β : Type u_3
                      inst✝¹ : LinearOrderedRing α
                      inst✝ : FloorRing α
                      a : α
                      n : Nat
                      ⊢ Iff (LE.le ((fun a => (Int.ceil a).toNat) a) n) (LE.le a ↑n)
                    -/
  gc_ceil a n := by rw [Int.toNat_le, Int.ceil_le, Int.cast_natCast]
                    /-
                      🎉 no goals
                    -/


theorem Int.floor_toNat (a : α) : ⌊a⌋.toNat = ⌊a⌋₊ :=
  rfl


theorem Int.ceil_toNat (a : α) : ⌈a⌉.toNat = ⌈a⌉₊ :=
  rfl


@[simp]
theorem Nat.floor_int : (Nat.floor : ℤ → ℕ) = Int.toNat :=
  rfl


@[simp]
theorem Nat.ceil_int : (Nat.ceil : ℤ → ℕ) = Int.toNat :=
  rfl


theorem Int.natCast_floor_eq_floor (ha : 0 ≤ a) : (⌊a⌋₊ : ℤ) = ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (↑(Nat.floor a)) (Int.floor a)
  -/
  rw [← Int.floor_toNat, Int.toNat_of_nonneg (Int.floor_nonneg.2 ha)]
  /-
    🎉 no goals
  -/


theorem Int.natCast_ceil_eq_ceil (ha : 0 ≤ a) : (⌈a⌉₊ : ℤ) = ⌈a⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le 0 a
    ⊢ Eq (↑(Nat.ceil a)) (Int.ceil a)
  -/
  rw [← Int.ceil_toNat, Int.toNat_of_nonneg (Int.ceil_nonneg ha)]
  /-
    🎉 no goals
  -/


theorem natCast_floor_eq_intCast_floor (ha : 0 ≤ a) : (⌊a⌋₊ : α) = ⌊a⌋ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le 0 a
    ⊢ Eq ↑(Nat.floor a) ↑(Int.floor a)
  -/
  rw [← Int.natCast_floor_eq_floor ha, Int.cast_natCast]
  /-
    🎉 no goals
  -/


theorem natCast_ceil_eq_intCast_ceil (ha : 0 ≤ a) : (⌈a⌉₊ : α) = ⌈a⌉ := by
  /-
    α : Type u_2
    inst✝¹ : LinearOrderedRing α
    inst✝ : FloorRing α
    a : α
    ha : LE.le 0 a
    ⊢ Eq ↑(Nat.ceil a) ↑(Int.ceil a)
  -/
  rw [← Int.natCast_ceil_eq_ceil ha, Int.cast_natCast]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-20")] alias Int.ofNat_floor_eq_floor := natCast_floor_eq_floor

@[deprecated (since := "2024-08-20")] alias Int.ofNat_ceil_eq_ceil := natCast_ceil_eq_ceil


/-- There exists at most one `FloorRing` structure on a given linear ordered ring. -/
theorem subsingleton_floorRing {α} [LinearOrderedRing α] : Subsingleton (FloorRing α) := by
  /-
    α : Type u_4
    inst✝ : LinearOrderedRing α
    ⊢ Subsingleton (FloorRing α)
  -/
  refine ⟨fun H₁ H₂ => ?_⟩
  have : H₁.floor = H₂.floor :=
    funext fun a => (H₁.gc_coe_floor.u_unique H₂.gc_coe_floor) fun _ => rfl
  /-
    α : Type u_4
    inst✝ : LinearOrderedRing α
    H₁ H₂ : FloorRing α
    this : Eq FloorRing.floor FloorRing.floor
    ⊢ Eq H₁ H₂
  -/
  have : H₁.ceil = H₂.ceil := funext fun a => (H₁.gc_ceil_coe.l_unique H₂.gc_ceil_coe) fun _ => rfl
  /-
    α : Type u_4
    inst✝ : LinearOrderedRing α
    H₁ H₂ : FloorRing α
    this✝ : Eq FloorRing.floor FloorRing.floor
    this : Eq FloorRing.ceil FloorRing.ceil
    ⊢ Eq H₁ H₂
  -/
  cases H₁; cases H₂; congr
                      /-
                        🎉 no goals
                      -/


private theorem int_floor_nonneg [LinearOrderedRing α] [FloorRing α] {a : α} (ha : 0 ≤ a) :
    0 ≤ ⌊a⌋ :=
  Int.floor_nonneg.2 ha


private theorem int_floor_nonneg_of_pos [LinearOrderedRing α] [FloorRing α] {a : α}
    (ha : 0 < a) :
    0 ≤ ⌊a⌋ :=
  int_floor_nonneg ha.le


/-- Extension for the `positivity` tactic: `Int.floor` is nonnegative if its input is. -/
@[positivity ⌊ _ ⌋]
def evalIntFloor : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℤ), ~q(@Int.floor $α' $i $j $a) =>
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa =>
        assertInstancesCommute
        pure (.nonnegative q(int_floor_nonneg_of_pos (α := $α') $pa))
    | .nonnegative pa =>
        assertInstancesCommute
        pure (.nonnegative q(int_floor_nonneg (α := $α') $pa))
    | _ => pure .none
  | _, _, _ => throwError "failed to match on Int.floor application"


private theorem nat_ceil_pos [LinearOrderedSemiring α] [FloorSemiring α] {a : α} :
    0 < a → 0 < ⌈a⌉₊ :=
  Nat.ceil_pos.2


/-- Extension for the `positivity` tactic: `Nat.ceil` is positive if its input is. -/
@[positivity ⌈ _ ⌉₊]
def evalNatCeil : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(@Nat.ceil $α' $i $j $a) =>
    let _i : Q(LinearOrderedSemiring $α') ← synthInstanceQ (u := u_1) _
    assertInstancesCommute
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa =>
      assertInstancesCommute
      pure (.positive q(nat_ceil_pos (α := $α') $pa))
    | _ => pure .none
  | _, _, _ => throwError "failed to match on Nat.ceil application"


private theorem int_ceil_pos [LinearOrderedRing α] [FloorRing α] {a : α} : 0 < a → 0 < ⌈a⌉ :=
  Int.ceil_pos.2


/-- Extension for the `positivity` tactic: `Int.ceil` is positive/nonnegative if its input is. -/
@[positivity ⌈ _ ⌉]
def evalIntCeil : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℤ), ~q(@Int.ceil $α' $i $j $a) =>
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa =>
        assertInstancesCommute
        pure (.positive q(int_ceil_pos (α := $α') $pa))
    | .nonnegative pa =>
        assertInstancesCommute
        pure (.nonnegative q(Int.ceil_nonneg (α := $α') $pa))
    | _ => pure .none
  | _, _, _ => throwError "failed to match on Int.ceil application"



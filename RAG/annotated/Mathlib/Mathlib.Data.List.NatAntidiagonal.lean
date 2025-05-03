/-- The antidiagonal of a natural number `n` is the list of pairs `(i, j)` such that `i + j = n`. -/
def antidiagonal (n : ℕ) : List (ℕ × ℕ) :=
  (range (n + 1)).map fun i ↦ (i, n - i)


/-- A pair (i, j) is contained in the antidiagonal of `n` if and only if `i + j = n`. -/
@[simp]
theorem mem_antidiagonal {n : ℕ} {x : ℕ × ℕ} : x ∈ antidiagonal n ↔ x.1 + x.2 = n := by
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem (List.Nat.antidiagonal n) x) (Eq (HAdd.hAdd x.1 x.2) n)
  -/
  rw [antidiagonal, mem_map]; constructor
    /-
      case mp
      n : Nat
      x : Prod Nat Nat
      ⊢ (Exists fun a => And (Membership.mem (List.range (HAdd.hAdd n 1)) a) (Eq { f …
    -/
  · rintro ⟨i, hi, rfl⟩
    /-
      case mp.intro.intro
      n i : Nat
      hi : Membership.mem (List.range (HAdd.hAdd n 1)) i
      ⊢ Eq (HAdd.hAdd { fst := i, snd := HSub.hSub n i }.1 { fst := i, snd := HSub.h …
    -/
    rw [mem_range, Nat.lt_succ_iff] at hi
    /-
      case mp.intro.intro
      n i : Nat
      hi : LE.le i n
      ⊢ Eq (HAdd.hAdd { fst := i, snd := HSub.hSub n i }.1 { fst := i, snd := HSub.h …
    -/
    exact Nat.add_sub_cancel' hi
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      x : Prod Nat Nat
      ⊢ Eq (HAdd.hAdd x.1 x.2) n → Exists fun a => And (Membership.mem (List.range ( …
    -/
  · rintro rfl
    /-
      case mpr
      x : Prod Nat Nat
      ⊢ Exists fun a => And (Membership.mem (List.range (HAdd.hAdd (HAdd.hAdd x.1 x. …
    -/
    refine ⟨x.fst, ?_, ?_⟩
      /-
        case mpr.refine_1
        x : Prod Nat Nat
        ⊢ Membership.mem (List.range (HAdd.hAdd (HAdd.hAdd x.1 x.2) 1)) x.1
      -/
    · rw [mem_range]
      /-
        case mpr.refine_1
        x : Prod Nat Nat
        ⊢ LT.lt x.1 (HAdd.hAdd (HAdd.hAdd x.1 x.2) 1)
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        x : Prod Nat Nat
        ⊢ Eq { fst := x.1, snd := HSub.hSub (HAdd.hAdd x.1 x.2) x.1 } x
      -/
    · exact Prod.ext rfl (by simp only [Nat.add_sub_cancel_left])
      /-
        🎉 no goals
      -/


/-- The length of the antidiagonal of `n` is `n + 1`. -/
@[simp]
theorem length_antidiagonal (n : ℕ) : (antidiagonal n).length = n + 1 := by
  /-
    n : Nat
    ⊢ Eq (List.Nat.antidiagonal n).length (HAdd.hAdd n 1)
  -/
  rw [antidiagonal, length_map, length_range]
  /-
    🎉 no goals
  -/


/-- The antidiagonal of `0` is the list `[(0, 0)]` -/
@[simp]
theorem antidiagonal_zero : antidiagonal 0 = [(0, 0)] :=
  rfl


/-- The antidiagonal of `n` does not contain duplicate entries. -/
theorem nodup_antidiagonal (n : ℕ) : Nodup (antidiagonal n) :=
  (nodup_range _).map ((@LeftInverse.injective ℕ (ℕ × ℕ) Prod.fst fun i ↦ (i, n - i)) fun _ ↦ rfl)


@[simp]
theorem antidiagonal_succ {n : ℕ} :
    antidiagonal (n + 1) = (0, n + 1) :: (antidiagonal n).map (Prod.map Nat.succ id) := by
  simp only [antidiagonal, range_succ_eq_map, map_cons, Nat.add_succ_sub_one,
    Nat.add_zero, id, eq_self_iff_true, Nat.sub_zero, map_map, Prod.map_apply]
  /-
    n : Nat
    ⊢ Eq (List.cons { fst := 0, snd := HAdd.hAdd n 1 } (List.cons { fst := Nat.suc …
  -/
  apply congr rfl (congr rfl _)
  /-
    n : Nat
    ⊢ Eq (List.map (Function.comp (fun i => { fst := i, snd := HSub.hSub (HAdd.hAd …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem antidiagonal_succ' {n : ℕ} :
    antidiagonal (n + 1) = (antidiagonal n).map (Prod.map id Nat.succ) ++ [(n + 1, 0)] := by
  simp only [antidiagonal, range_succ, Nat.add_sub_cancel_left, map_append, append_assoc,
    Nat.sub_self, singleton_append, map_map, map]
  /-
    n : Nat
    ⊢ Eq (HAppend.hAppend (List.map (fun i => { fst := i, snd := HSub.hSub (HAdd.h …
  -/
  congr 1
  /-
    case e_a
    n : Nat
    ⊢ Eq (List.map (fun i => { fst := i, snd := HSub.hSub (HAdd.hAdd n 1) i }) (Li …
  -/
  apply map_congr_left
  /-
    case e_a.h
    n : Nat
    ⊢ ∀ (a : Nat), Membership.mem (List.range n) a → Eq { fst := a, snd := HSub.hS …
  -/
  simp +contextual [le_of_lt, Nat.sub_add_comm]
  /-
    🎉 no goals
  -/


theorem antidiagonal_succ_succ' {n : ℕ} :
    antidiagonal (n + 2) =
      (0, n + 2) :: (antidiagonal n).map (Prod.map Nat.succ Nat.succ) ++ [(n + 2, 0)] := by
  /-
    n : Nat
    ⊢ Eq (List.Nat.antidiagonal (HAdd.hAdd n 2)) (HAppend.hAppend (List.cons { fst …
  -/
  rw [antidiagonal_succ']
  simp only [antidiagonal_succ, map_cons, Prod.map_apply, id_eq, map_map, cons_append, cons.injEq,
    append_cancel_right_eq, true_and]
  /-
    n : Nat
    ⊢ Eq (List.map (Function.comp (Prod.map id Nat.succ) (Prod.map Nat.succ id)) ( …
  -/
  ext
  /-
    case h.a
    n n✝ : Nat
    a✝ : Prod Nat Nat
    ⊢ Iff (Membership.mem (GetElem?.getElem? (List.map (Function.comp (Prod.map id …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem map_swap_antidiagonal {n : ℕ} :
    (antidiagonal n).map Prod.swap = (antidiagonal n).reverse := by
  rw [antidiagonal, map_map, ← List.map_reverse, range_eq_range', reverse_range', ←
    range_eq_range', map_map]
  /-
    n : Nat
    ⊢ Eq (List.map (Function.comp Prod.swap fun i => { fst := i, snd := HSub.hSub  …
  -/
  apply map_congr_left
  /-
    case h
    n : Nat
    ⊢ ∀ (a : Nat), Membership.mem (List.range (HAdd.hAdd n 1)) a → Eq (Function.co …
  -/
  simp +contextual [Nat.sub_sub_self, Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/



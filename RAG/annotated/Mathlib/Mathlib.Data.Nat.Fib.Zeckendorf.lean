@[nolint defLemma docBlame]
local instance : IsTrans ℕ fun a b ↦ b + 2 ≤ a where
  trans _a _b _c hba hcb := hcb.trans <| le_self_add.trans hba


/-- A list of natural numbers is a Zeckendorf representation (of a natural number) if it is an
increasing sequence of non-consecutive numbers greater than or equal to `2`.

This is relevant for Zeckendorf's theorem, since if we write a natural `n` as a sum of Fibonacci
numbers `(l.map fib).sum`, `IsZeckendorfRep l` exactly means that we can't simplify any expression
of the form `fib n + fib (n + 1) = fib (n + 2)`, `fib 1 = fib 2` or `fib 0 = 0` in the sum. -/
def IsZeckendorfRep (l : List ℕ) : Prop := (l ++ [0]).Chain' (fun a b ↦ b + 2 ≤ a)


@[simp]
                                                     /-
                                                       ⊢ List.nil.IsZeckendorfRep
                                                     -/
lemma IsZeckendorfRep_nil : IsZeckendorfRep [] := by simp [IsZeckendorfRep]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma IsZeckendorfRep.sum_fib_lt : ∀ {n l}, IsZeckendorfRep l → (∀ a ∈ (l ++ [0]).head?, a < n) →
    (l.map fib).sum < fib n
  | _, [], _, hn => fib_pos.2 <| hn _ rfl
  | n, a :: l, hl, hn => by
    /-
      n a : Nat
      l : List Nat
      hl : (List.cons a l).IsZeckendorfRep
      hn : ∀ (a_1 : Nat), Membership.mem (HAppend.hAppend (List.cons a l) (List.cons …
      ⊢ LT.lt (List.map Nat.fib (List.cons a l)).sum (Nat.fib n)
    -/
    simp only [IsZeckendorfRep, cons_append, chain'_iff_pairwise, pairwise_cons] at hl
    have : ∀ b, b ∈ head? (l ++ [0]) → b < a - 1 :=
      fun b hb ↦ lt_tsub_iff_right.2 <| hl.1 _ <| mem_of_mem_head? hb
    simp only [mem_append, mem_singleton, ← chain'_iff_pairwise, or_imp, forall_and, forall_eq,
      zero_add] at hl
    /-
      n a : Nat
      l : List Nat
      hn : ∀ (a_1 : Nat), Membership.mem (HAppend.hAppend (List.cons a l) (List.cons …
      this : ∀ (b : Nat), Membership.mem (HAppend.hAppend l (List.cons 0 List.nil)). …
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ LT.lt (List.map Nat.fib (List.cons a l)).sum (Nat.fib n)
    -/
    simp only [map, List.sum_cons]
    /-
      n a : Nat
      l : List Nat
      hn : ∀ (a_1 : Nat), Membership.mem (HAppend.hAppend (List.cons a l) (List.cons …
      this : ∀ (b : Nat), Membership.mem (HAppend.hAppend l (List.cons 0 List.nil)). …
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ LT.lt (HAdd.hAdd (Nat.fib a) (List.map Nat.fib l).sum) (Nat.fib n)
    -/
    refine (add_lt_add_left (sum_fib_lt hl.2 this) _).trans_le ?_
    /-
      n a : Nat
      l : List Nat
      hn : ∀ (a_1 : Nat), Membership.mem (HAppend.hAppend (List.cons a l) (List.cons …
      this : ∀ (b : Nat), Membership.mem (HAppend.hAppend l (List.cons 0 List.nil)). …
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ LE.le (HAdd.hAdd (Nat.fib a) (Nat.fib (HSub.hSub a 1))) (Nat.fib n)
    -/
    rw [add_comm, ← fib_add_one (hl.1.2.trans_lt' zero_lt_two).ne']
    /-
      n a : Nat
      l : List Nat
      hn : ∀ (a_1 : Nat), Membership.mem (HAppend.hAppend (List.cons a l) (List.cons …
      this : ∀ (b : Nat), Membership.mem (HAppend.hAppend l (List.cons 0 List.nil)). …
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ LE.le (Nat.fib (HAdd.hAdd a 1)) (Nat.fib n)
    -/
    exact fib_mono (hn _ rfl)
    /-
      🎉 no goals
    -/


/-- The greatest index of a Fibonacci number less than or equal to `n`. -/
def greatestFib (n : ℕ) : ℕ := (n + 1).findGreatest (fun k ↦ fib k ≤ n)


lemma fib_greatestFib_le (n : ℕ) : fib (greatestFib n) ≤ n :=
  findGreatest_spec (P := (fun k ↦ fib k ≤ n)) (zero_le _) <| zero_le _


lemma greatestFib_mono : Monotone greatestFib :=
  fun _a _b hab ↦ findGreatest_mono (fun _k ↦ hab.trans') <| add_le_add_right hab _


@[simp] lemma le_greatestFib : m ≤ greatestFib n ↔ fib m ≤ n :=
  ⟨fun h ↦ (fib_mono h).trans <| fib_greatestFib_le _,
    fun h ↦ le_findGreatest (m.le_fib_add_one.trans <| add_le_add_right h _) h⟩


@[simp] lemma greatestFib_lt : greatestFib m < n ↔ m < fib n :=
  lt_iff_lt_of_le_iff_le le_greatestFib


lemma lt_fib_greatestFib_add_one (n : ℕ) : n < fib (greatestFib n + 1) :=
  greatestFib_lt.1 <| lt_succ_self _


@[simp] lemma greatestFib_fib : ∀ {n}, n ≠ 1 → greatestFib (fib n) = n
  | 0, _ => rfl
  | _n + 2, _ => findGreatest_eq_iff.2
    ⟨le_fib_add_one _, fun _ ↦ le_rfl, fun _m hnm _ ↦ ((fib_lt_fib le_add_self).2 hnm).not_le⟩


@[simp] lemma greatestFib_eq_zero : greatestFib n = 0 ↔ n = 0 :=
              /-
                n : Nat
                h : Eq n.greatestFib 0
                ⊢ Eq n 0
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by simpa using findGreatest_eq_zero_iff.1 h zero_lt_one le_add_self, by rintro rfl; rfl⟩
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


lemma greatestFib_ne_zero : greatestFib n ≠ 0 ↔ n ≠ 0 := greatestFib_eq_zero.not


                                                                /-
                                                                  n : Nat
                                                                  ⊢ Iff (LT.lt 0 n.greatestFib) (LT.lt 0 n)
                                                                -/
@[simp] lemma greatestFib_pos : 0 < greatestFib n ↔ 0 < n := by simp [pos_iff_ne_zero]
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma greatestFib_sub_fib_greatestFib_le_greatestFib (hn : n ≠ 0) :
    greatestFib (n - fib (greatestFib n)) ≤ greatestFib n - 2 := by
  rw [← Nat.lt_succ_iff, greatestFib_lt, tsub_lt_iff_right n.fib_greatestFib_le, Nat.sub_succ,
    succ_pred, ← fib_add_one]
    /-
      n : Nat
      hn : Ne n 0
      ⊢ LT.lt n (Nat.fib (HAdd.hAdd n.greatestFib 1))
    -/
  · exact n.lt_fib_greatestFib_add_one
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      hn : Ne n 0
      ⊢ Ne n.greatestFib 0
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      hn : Ne n 0
      ⊢ Ne (HSub.hSub n.greatestFib 1) 0
    -/
  · simpa [← succ_le_iff, tsub_eq_zero_iff_le] using hn.bot_lt
    /-
      🎉 no goals
    -/


private lemma zeckendorf_aux (hm : 0 < m) : m - fib (greatestFib m) < m :=
tsub_lt_self hm <| fib_pos.2 <| findGreatest_pos.2 ⟨1, zero_lt_one, le_add_self, hm⟩


/-- The Zeckendorf representation of a natural number.

Note: For unfolding, you should use the equational lemmas `Nat.zeckendorf_zero` and
`Nat.zeckendorf_of_pos` instead of the autogenerated one. -/
def zeckendorf : ℕ → List ℕ
  | 0 => []
  | m@(_ + 1) =>
    letI a := greatestFib m
    a :: zeckendorf (m - fib a)
/-
  m n✝ : Nat
  h✝ : Eq m (HAdd.hAdd n✝ 1)
  ⊢ LT.lt (HSub.hSub m (Nat.fib m.greatestFib)) (namedPattern m n✝.succ h✝)
-/
decreasing_by simp_wf; subst_vars; apply zeckendorf_aux (zero_lt_succ _)
/-
  🎉 no goals
-/



@[simp] lemma zeckendorf_zero : zeckendorf 0 = [] := zeckendorf.eq_1 ..


@[simp] lemma zeckendorf_succ (n : ℕ) :
    zeckendorf (n + 1) = greatestFib (n + 1) :: zeckendorf (n + 1 - fib (greatestFib (n + 1))) :=
  zeckendorf.eq_2 ..


@[simp] lemma zeckendorf_of_pos : ∀ {n}, 0 < n →
    zeckendorf n = greatestFib n :: zeckendorf (n - fib (greatestFib n))
  | _n + 1, _ => zeckendorf_succ _


lemma isZeckendorfRep_zeckendorf : ∀ n, (zeckendorf n).IsZeckendorfRep
            /-
              ⊢ (Nat.zeckendorf 0).IsZeckendorfRep
            -/
  | 0 => by simp only [zeckendorf_zero, IsZeckendorfRep_nil]
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      n : Nat
      ⊢ (HAdd.hAdd n 1).zeckendorf.IsZeckendorfRep
    -/
    rw [zeckendorf_succ, IsZeckendorfRep, List.cons_append]
    /-
      n : Nat
      ⊢ List.Chain' (fun a b => LE.le (HAdd.hAdd b 2) a) (List.cons (HAdd.hAdd n 1). …
    -/
    refine (isZeckendorfRep_zeckendorf _).cons' (fun a ha ↦ ?_)
    /-
      n a : Nat
      ha : Membership.mem (HAppend.hAppend (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd …
      ⊢ LE.le (HAdd.hAdd a 2) (HAdd.hAdd n 1).greatestFib
    -/
    obtain h | h := eq_zero_or_pos (n + 1 - fib (greatestFib (n + 1)))
      /-
        case inl
        n a : Nat
        ha : Membership.mem (HAppend.hAppend (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd …
        h : Eq (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)) 0
        ⊢ LE.le (HAdd.hAdd a 2) (HAdd.hAdd n 1).greatestFib
      -/
    · simp only [h, zeckendorf_zero, nil_append, head?_cons, Option.mem_some_iff] at ha
      /-
        case inl
        n a : Nat
        h : Eq (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)) 0
        ha : Eq 0 a
        ⊢ LE.le (HAdd.hAdd a 2) (HAdd.hAdd n 1).greatestFib
      -/
      subst ha
      /-
        case inl
        n : Nat
        h : Eq (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)) 0
        ⊢ LE.le (HAdd.hAdd 0 2) (HAdd.hAdd n 1).greatestFib
      -/
      exact le_greatestFib.2 le_add_self
      /-
        🎉 no goals
      -/
    /-
      case inr
      n a : Nat
      ha : Membership.mem (HAppend.hAppend (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd …
      h : GT.gt (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)) 0
      ⊢ LE.le (HAdd.hAdd a 2) (HAdd.hAdd n 1).greatestFib
    -/
    rw [zeckendorf_of_pos h, cons_append, head?_cons, Option.mem_some_iff] at ha
    /-
      case inr
      n a : Nat
      ha : Eq (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)).grea …
      h : GT.gt (HSub.hSub (HAdd.hAdd n 1) (Nat.fib (HAdd.hAdd n 1).greatestFib)) 0
      ⊢ LE.le (HAdd.hAdd a 2) (HAdd.hAdd n 1).greatestFib
    -/
    subst a
    exact add_le_of_le_tsub_right_of_le (le_greatestFib.2 le_add_self)
      (greatestFib_sub_fib_greatestFib_le_greatestFib n.succ_ne_zero)


lemma zeckendorf_sum_fib : ∀ {l}, IsZeckendorfRep l → zeckendorf (l.map fib).sum = l
                /-
                  x✝ : List.nil.IsZeckendorfRep
                  ⊢ Eq (List.map Nat.fib List.nil).sum.zeckendorf List.nil
                -/
  | [], _ => by simp only [map_nil, List.sum_nil, zeckendorf_zero]
                /-
                  🎉 no goals
                -/
  | a :: l, hl => by
    /-
      a : Nat
      l : List Nat
      hl : (List.cons a l).IsZeckendorfRep
      ⊢ Eq (List.map Nat.fib (List.cons a l)).sum.zeckendorf (List.cons a l)
    -/
    have hl' := hl
    simp only [IsZeckendorfRep, cons_append, chain'_iff_pairwise, pairwise_cons, mem_append,
      mem_singleton, or_imp, forall_and, forall_eq, zero_add] at hl
    /-
      a : Nat
      l : List Nat
      hl' : (List.cons a l).IsZeckendorfRep
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ Eq (List.map Nat.fib (List.cons a l)).sum.zeckendorf (List.cons a l)
    -/
    rw [← chain'_iff_pairwise] at hl
    /-
      a : Nat
      l : List Nat
      hl' : (List.cons a l).IsZeckendorfRep
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ⊢ Eq (List.map Nat.fib (List.cons a l)).sum.zeckendorf (List.cons a l)
    -/
    have ha : 0 < a := hl.1.2.trans_lt' zero_lt_two
    suffices h : greatestFib (fib a + sum (map fib l)) = a by
      simp only [map, List.sum_cons, add_pos_iff, fib_pos.2 ha, true_or, zeckendorf_of_pos, h,
      add_tsub_cancel_left, zeckendorf_sum_fib hl.2]
    simp only [add_comm, add_assoc, greatestFib, findGreatest_eq_iff, ne_eq, ha.ne',
      not_false_eq_true, le_add_iff_nonneg_left, _root_.zero_le, forall_true_left, not_le, true_and]
    /-
      a : Nat
      l : List Nat
      hl' : (List.cons a l).IsZeckendorfRep
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ha : LT.lt 0 a
      ⊢ And (LE.le a (HAdd.hAdd (List.map Nat.fib l).sum (HAdd.hAdd (Nat.fib a) 1))) …
    -/
    refine ⟨le_add_of_le_right <| le_fib_add_one _, fun n hn _ ↦ ?_⟩
    /-
      a : Nat
      l : List Nat
      hl' : (List.cons a l).IsZeckendorfRep
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ha : LT.lt 0 a
      n : Nat
      hn : LT.lt a n
      x✝ : LE.le n (HAdd.hAdd (List.map Nat.fib l).sum (HAdd.hAdd (Nat.fib a) 1))
      ⊢ LT.lt (HAdd.hAdd (List.map Nat.fib l).sum (Nat.fib a)) (Nat.fib n)
    -/
    rw [add_comm, ← List.sum_cons, ← map_cons]
    /-
      a : Nat
      l : List Nat
      hl' : (List.cons a l).IsZeckendorfRep
      hl : And (And (∀ (x : Nat), Membership.mem l x → LE.le (HAdd.hAdd x 2) a) (LE. …
      ha : LT.lt 0 a
      n : Nat
      hn : LT.lt a n
      x✝ : LE.le n (HAdd.hAdd (List.map Nat.fib l).sum (HAdd.hAdd (Nat.fib a) 1))
      ⊢ LT.lt (List.map Nat.fib (List.cons a l)).sum (Nat.fib n)
    -/
    exact hl'.sum_fib_lt (by simpa)
    /-
      🎉 no goals
    -/


@[simp] lemma sum_zeckendorf_fib (n : ℕ) : (n.zeckendorf.map fib).sum = n := by
  /-
    n : Nat
    ⊢ Eq (List.map Nat.fib n.zeckendorf).sum n
  -/
                                          /-
                                            🎉 no goals
                                          -/
  induction n using zeckendorf.induct <;> simp_all [fib_greatestFib_le]
                                          /-
                                            🎉 no goals
                                          -/


/-- **Zeckendorf's Theorem** as an equivalence between natural numbers and Zeckendorf
representations. Every natural number can be written uniquely as a sum of non-consecutive Fibonacci
numbers (if we forget about the first two terms `F₀ = 0`, `F₁ = 1`). -/
def zeckendorfEquiv : ℕ ≃ {l // IsZeckendorfRep l} where
  toFun n := ⟨zeckendorf n, isZeckendorfRep_zeckendorf _⟩
  invFun l := (map fib l).sum
  left_inv := sum_zeckendorf_fib
  right_inv l := Subtype.ext <| zeckendorf_sum_fib l.2



/-- `List.antidiagonalTuple k n` is a list of all `k`-tuples which sum to `n`.

This list contains no duplicates (`List.Nat.nodup_antidiagonalTuple`), and is sorted
lexicographically (`List.Nat.antidiagonalTuple_pairwise_pi_lex`), starting with `![0, ..., n]`
and ending with `![n, ..., 0]`.

```
#eval antidiagonalTuple 3 2
-- [![0, 0, 2], ![0, 1, 1], ![0, 2, 0], ![1, 0, 1], ![1, 1, 0], ![2, 0, 0]]
```
-/
def antidiagonalTuple : ∀ k, ℕ → List (Fin k → ℕ)
  | 0, 0 => [![]]
  | 0, _ + 1 => []
  | k + 1, n =>
    (List.Nat.antidiagonal n).flatMap fun ni =>
      (antidiagonalTuple k ni.2).map fun x => Fin.cons ni.1 x


@[simp]
theorem antidiagonalTuple_zero_zero : antidiagonalTuple 0 0 = [![]] :=
  rfl


@[simp]
theorem antidiagonalTuple_zero_succ (n : ℕ) : antidiagonalTuple 0 (n + 1) = [] :=
  rfl


theorem mem_antidiagonalTuple {n : ℕ} {k : ℕ} {x : Fin k → ℕ} :
    x ∈ antidiagonalTuple k n ↔ ∑ i, x i = n := by
  induction x using Fin.consInduction generalizing n with
  | h0 =>
    cases n
    · decide
    · simp [eq_comm]
  | h x₀ x ih =>
    simp_rw [Fin.sum_cons, antidiagonalTuple, List.mem_flatMap, List.mem_map,
      List.Nat.mem_antidiagonal, Fin.cons_eq_cons, exists_eq_right_right, ih,
      @eq_comm _ _ (Prod.snd _), and_comm (a := Prod.snd _ = _),
      ← Prod.mk.inj_iff (a₁ := Prod.fst _), exists_eq_right]


/-- The antidiagonal of `n` does not contain duplicate entries. -/
theorem nodup_antidiagonalTuple (k n : ℕ) : List.Nodup (antidiagonalTuple k n) := by
  /-
    k n : Nat
    ⊢ (List.Nat.antidiagonalTuple k n).Nodup
  -/
  induction' k with k ih generalizing n
    /-
      case zero
      n : Nat
      ⊢ (List.Nat.antidiagonalTuple 0 n).Nodup
    -/
  · cases n
      /-
        case zero.zero
        ⊢ (List.Nat.antidiagonalTuple 0 0).Nodup
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case zero.succ
        n✝ : Nat
        ⊢ (List.Nat.antidiagonalTuple 0 (HAdd.hAdd n✝ 1)).Nodup
      -/
    · simp [eq_comm]
      /-
        🎉 no goals
      -/
  /-
    case succ
    k : Nat
    ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
    n : Nat
    ⊢ (List.Nat.antidiagonalTuple (HAdd.hAdd k 1) n).Nodup
  -/
  simp_rw [antidiagonalTuple, List.nodup_flatMap]
  /-
    case succ
    k : Nat
    ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
    n : Nat
    ⊢ And (∀ (x : Prod Nat Nat), Membership.mem (List.Nat.antidiagonal n) x → (Lis …
  -/
  constructor
    /-
      case succ.left
      k : Nat
      ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
      n : Nat
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (List.Nat.antidiagonal n) x → (List.map …
    -/
  · intro i _
    /-
      case succ.left
      k : Nat
      ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
      n : Nat
      i : Prod Nat Nat
      a✝ : Membership.mem (List.Nat.antidiagonal n) i
      ⊢ (List.map (fun x => Fin.cons i.1 x) (List.Nat.antidiagonalTuple k i.2)).Nodup
    -/
    exact (ih i.snd).map (Fin.cons_right_injective (α := fun _ => ℕ) i.fst)
    /-
      🎉 no goals
    -/
  /-
    case succ.right
    k : Nat
    ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
    n : Nat
    ⊢ List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x => Fin …
  -/
  induction' n with n n_ih
    /-
      case succ.right.zero
      k : Nat
      ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
      ⊢ List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x => Fin …
    -/
  · exact List.pairwise_singleton _ _
    /-
      🎉 no goals
    -/
    /-
      case succ.right.succ
      k : Nat
      ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
      n : Nat
      n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
      ⊢ List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x => Fin …
    -/
  · rw [List.Nat.antidiagonal_succ]
    /-
      case succ.right.succ
      k : Nat
      ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
      n : Nat
      n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
      ⊢ List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x => Fin …
    -/
    refine List.Pairwise.cons (fun a ha x hx₁ hx₂ => ?_) (n_ih.map _ fun a b h x hx₁ hx₂ => ?_)
      /-
        case succ.right.succ.refine_1
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a : Prod Nat Nat
        ha : Membership.mem (List.map (Prod.map Nat.succ id) (List.Nat.antidiagonal n) …
        x : Fin (HAdd.hAdd k 1) → Nat
        hx₁ : Membership.mem ((fun ni => List.map (fun x => Fin.cons ni.1 x) (List.Nat …
        hx₂ : Membership.mem ((fun ni => List.map (fun x => Fin.cons ni.1 x) (List.Nat …
        ⊢ False
      -/
    · rw [List.mem_map] at hx₁ hx₂ ha
      /-
        case succ.right.succ.refine_1
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a : Prod Nat Nat
        ha : Exists fun a_1 => And (Membership.mem (List.Nat.antidiagonal n) a_1) (Eq  …
        x : Fin (HAdd.hAdd k 1) → Nat
        hx₁ : Exists fun a => And (Membership.mem (List.Nat.antidiagonalTuple k { fst  …
        hx₂ : Exists fun a_1 => And (Membership.mem (List.Nat.antidiagonalTuple k a.2) …
        ⊢ False
      -/
      obtain ⟨⟨a, -, rfl⟩, ⟨x₁, -, rfl⟩, ⟨x₂, -, h⟩⟩ := ha, hx₁, hx₂
      /-
        case succ.right.succ.refine_1.intro.intro.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a : Prod Nat Nat
        x₁ x₂ : Fin k → Nat
        h : Eq (Fin.cons (Prod.map Nat.succ id a).1 x₂) (Fin.cons { fst := 0, snd := H …
        ⊢ False
      -/
      rw [Fin.cons_eq_cons] at h
      /-
        case succ.right.succ.refine_1.intro.intro.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a : Prod Nat Nat
        x₁ x₂ : Fin k → Nat
        h : And (Eq (Prod.map Nat.succ id a).1 { fst := 0, snd := HAdd.hAdd n 1 }.1) ( …
        ⊢ False
      -/
      injection h.1
      /-
        🎉 no goals
      -/
      /-
        case succ.right.succ.refine_2
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x : Fin (HAdd.hAdd k 1) → Nat
        hx₁ : Membership.mem ((fun ni => List.map (fun x => Fin.cons ni.1 x) (List.Nat …
        hx₂ : Membership.mem ((fun ni => List.map (fun x => Fin.cons ni.1 x) (List.Nat …
        ⊢ False
      -/
    · rw [List.mem_map] at hx₁ hx₂
      /-
        case succ.right.succ.refine_2
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x : Fin (HAdd.hAdd k 1) → Nat
        hx₁ : Exists fun a_1 => And (Membership.mem (List.Nat.antidiagonalTuple k (Pro …
        hx₂ : Exists fun a => And (Membership.mem (List.Nat.antidiagonalTuple k (Prod. …
        ⊢ False
      -/
      obtain ⟨⟨x₁, hx₁, rfl⟩, ⟨x₂, hx₂, h₁₂⟩⟩ := hx₁, hx₂
      /-
        case succ.right.succ.refine_2.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x₁ : Fin k → Nat
        hx₁ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id a).2) …
        x₂ : Fin k → Nat
        hx₂ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id b).2) …
        h₁₂ : Eq (Fin.cons (Prod.map Nat.succ id b).1 x₂) (Fin.cons (Prod.map Nat.succ …
        ⊢ False
      -/
      dsimp at h₁₂
      /-
        case succ.right.succ.refine_2.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x₁ : Fin k → Nat
        hx₁ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id a).2) …
        x₂ : Fin k → Nat
        hx₂ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id b).2) …
        h₁₂ : Eq (Fin.cons (HAdd.hAdd b.1 1) x₂) (Fin.cons (HAdd.hAdd a.1 1) x₁)
        ⊢ False
      -/
      rw [Fin.cons_eq_cons, Nat.succ_inj'] at h₁₂
      /-
        case succ.right.succ.refine_2.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x₁ : Fin k → Nat
        hx₁ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id a).2) …
        x₂ : Fin k → Nat
        hx₂ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id b).2) …
        h₁₂ : And (Eq b.1 a.1) (Eq x₂ x₁)
        ⊢ False
      -/
      obtain ⟨h₁₂, rfl⟩ := h₁₂
      /-
        case succ.right.succ.refine_2.intro.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : Function.onFun List.Disjoint (fun ni => List.map (fun x => Fin.cons ni.1 x …
        x₂ : Fin k → Nat
        hx₂ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id b).2) …
        h₁₂ : Eq b.1 a.1
        hx₁ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id a).2) …
        ⊢ False
      -/
      rw [Function.onFun, h₁₂] at h
      /-
        case succ.right.succ.refine_2.intro.intro.intro.intro.intro
        k : Nat
        ih : ∀ (n : Nat), (List.Nat.antidiagonalTuple k n).Nodup
        n : Nat
        n_ih : List.Pairwise (Function.onFun List.Disjoint fun ni => List.map (fun x = …
        a b : Prod Nat Nat
        h : (List.map (fun x => Fin.cons a.1 x) (List.Nat.antidiagonalTuple k a.2)).Di …
        x₂ : Fin k → Nat
        hx₂ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id b).2) …
        h₁₂ : Eq b.1 a.1
        hx₁ : Membership.mem (List.Nat.antidiagonalTuple k (Prod.map Nat.succ id a).2) …
        ⊢ False
      -/
      exact h (List.mem_map_of_mem _ hx₁) (List.mem_map_of_mem _ hx₂)
      /-
        🎉 no goals
      -/


theorem antidiagonalTuple_zero_right : ∀ k, antidiagonalTuple k 0 = [0]
  | 0 => (congr_arg fun x => [x]) <| Subsingleton.elim _ _
  | k + 1 => by
    rw [antidiagonalTuple, antidiagonal_zero, List.flatMap_singleton,
      antidiagonalTuple_zero_right k, List.map_singleton]
    /-
      k : Nat
      ⊢ Eq (List.cons (Fin.cons { fst := 0, snd := 0 }.1 0) List.nil) (List.cons 0 L …
    -/
    exact congr_arg (fun x => [x]) Matrix.cons_zero_zero
    /-
      🎉 no goals
    -/


@[simp]
theorem antidiagonalTuple_one (n : ℕ) : antidiagonalTuple 1 n = [![n]] := by
  simp_rw [antidiagonalTuple, antidiagonal, List.range_succ, List.map_append, List.map_singleton,
    Nat.sub_self, List.flatMap_append, List.flatMap_singleton, List.flatMap_map]
  /-
    n : Nat
    ⊢ Eq (HAppend.hAppend ((List.range n).flatMap fun a => List.map (fun x => Fin. …
  -/
  conv_rhs => rw [← List.nil_append [![n]]]
  /-
    n : Nat
    ⊢ Eq (HAppend.hAppend ((List.range n).flatMap fun a => List.map (fun x => Fin. …
  -/
  congr 1
  /-
    case e_a
    n : Nat
    ⊢ Eq ((List.range n).flatMap fun a => List.map (fun x => Fin.cons a x) (List.N …
  -/
  simp_rw [List.flatMap_eq_nil_iff, List.mem_range, List.map_eq_nil_iff]
  /-
    case e_a
    n : Nat
    ⊢ ∀ (x : Nat), LT.lt x n → Eq (List.Nat.antidiagonalTuple 0 (HSub.hSub n x)) L …
  -/
  intro x hx
  /-
    case e_a
    n x : Nat
    hx : LT.lt x n
    ⊢ Eq (List.Nat.antidiagonalTuple 0 (HSub.hSub n x)) List.nil
  -/
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_lt hx
  /-
    case e_a.intro
    x m : Nat
    hx : LT.lt x (HAdd.hAdd (HAdd.hAdd x m) 1)
    ⊢ Eq (List.Nat.antidiagonalTuple 0 (HSub.hSub (HAdd.hAdd (HAdd.hAdd x m) 1) x) …
  -/
  rw [add_assoc, add_tsub_cancel_left, antidiagonalTuple_zero_succ]
  /-
    🎉 no goals
  -/


theorem antidiagonalTuple_two (n : ℕ) :
    antidiagonalTuple 2 n = (antidiagonal n).map fun i => ![i.1, i.2] := by
  /-
    n : Nat
    ⊢ Eq (List.Nat.antidiagonalTuple 2 n) (List.map (fun i => Matrix.vecCons i.1 ( …
  -/
  rw [antidiagonalTuple]
  /-
    n : Nat
    ⊢ Eq ((List.Nat.antidiagonal n).flatMap fun ni => List.map (fun x => Fin.cons  …
  -/
  simp_rw [antidiagonalTuple_one, List.map_singleton]
  /-
    n : Nat
    ⊢ Eq ((List.Nat.antidiagonal n).flatMap fun ni => List.cons (Fin.cons ni.1 (Ma …
  -/
  rw [List.map_eq_flatMap]
  /-
    n : Nat
    ⊢ Eq ((List.Nat.antidiagonal n).flatMap fun ni => List.cons (Fin.cons ni.1 (Ma …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem antidiagonalTuple_pairwise_pi_lex :
    ∀ k n, (antidiagonalTuple k n).Pairwise (Pi.Lex (· < ·) @fun _ => (· < ·))
  | 0, 0 => List.pairwise_singleton _ _
  | 0, _ + 1 => List.Pairwise.nil
  | k + 1, n => by
    simp_rw [antidiagonalTuple, List.pairwise_flatMap, List.pairwise_map, List.mem_map,
      forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    /-
      k n : Nat
      ⊢ And (∀ (a : Prod Nat Nat), Membership.mem (List.Nat.antidiagonal n) a → List …
    -/
    simp only [mem_antidiagonal, Prod.forall, and_imp, forall_apply_eq_imp_iff₂]
    simp only [Fin.pi_lex_lt_cons_cons, eq_self_iff_true, true_and, lt_self_iff_false,
      false_or]
    /-
      k n : Nat
      ⊢ And (∀ (a b : Nat), Eq (HAdd.hAdd a b) n → List.Pairwise (fun a b => Pi.Lex  …
    -/
    refine ⟨fun _ _ _ => antidiagonalTuple_pairwise_pi_lex k _, ?_⟩
    /-
      k n : Nat
      ⊢ List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.Nat.an …
    -/
    induction' n with n n_ih
      /-
        case zero
        k : Nat
        ⊢ List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
    · rw [antidiagonal_zero]
      /-
        case zero
        k : Nat
        ⊢ List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
      exact List.pairwise_singleton _ _
      /-
        🎉 no goals
      -/
      /-
        case succ
        k n : Nat
        n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
        ⊢ List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
    · rw [antidiagonal_succ, List.pairwise_cons, List.pairwise_map]
      /-
        case succ
        k n : Nat
        n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
        ⊢ And (∀ (a' : Prod Nat Nat), Membership.mem (List.map (Prod.map Nat.succ id)  …
      -/
      refine ⟨fun p hp x hx y hy => ?_, ?_⟩
        /-
          case succ.refine_1
          k n : Nat
          n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
          p : Prod Nat Nat
          hp : Membership.mem (List.map (Prod.map Nat.succ id) (List.Nat.antidiagonal n) …
          x : Fin k → Nat
          hx : Membership.mem (List.Nat.antidiagonalTuple k { fst := 0, snd := HAdd.hAdd …
          y : Fin k → Nat
          hy : Membership.mem (List.Nat.antidiagonalTuple k p.2) y
          ⊢ Or (LT.lt { fst := 0, snd := HAdd.hAdd n 1 }.1 p.1) (And (Eq { fst := 0, snd …
        -/
      · rw [List.mem_map, Prod.exists] at hp
        /-
          case succ.refine_1
          k n : Nat
          n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
          p : Prod Nat Nat
          hp : Exists fun a => Exists fun b => And (Membership.mem (List.Nat.antidiagona …
          x : Fin k → Nat
          hx : Membership.mem (List.Nat.antidiagonalTuple k { fst := 0, snd := HAdd.hAdd …
          y : Fin k → Nat
          hy : Membership.mem (List.Nat.antidiagonalTuple k p.2) y
          ⊢ Or (LT.lt { fst := 0, snd := HAdd.hAdd n 1 }.1 p.1) (And (Eq { fst := 0, snd …
        -/
        obtain ⟨a, b, _, rfl : (Nat.succ a, b) = p⟩ := hp
        /-
          case succ.refine_1.intro.intro.intro
          k n : Nat
          n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
          x : Fin k → Nat
          hx : Membership.mem (List.Nat.antidiagonalTuple k { fst := 0, snd := HAdd.hAdd …
          y : Fin k → Nat
          a b : Nat
          left✝ : Membership.mem (List.Nat.antidiagonal n) { fst := a, snd := b }
          hy : Membership.mem (List.Nat.antidiagonalTuple k { fst := a.succ, snd := b }. …
          ⊢ Or (LT.lt { fst := 0, snd := HAdd.hAdd n 1 }.1 { fst := a.succ, snd := b }.1 …
        -/
        exact Or.inl (Nat.zero_lt_succ _)
        /-
          🎉 no goals
        -/
      /-
        case succ.refine_2
        k n : Nat
        n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
        ⊢ List.Pairwise (fun a b => ∀ (a_1 : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
      dsimp
      /-
        case succ.refine_2
        k n : Nat
        n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
        ⊢ List.Pairwise (fun a b => ∀ (a_1 : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
      simp_rw [Nat.succ_inj', Nat.succ_lt_succ_iff]
      /-
        case succ.refine_2
        k n : Nat
        n_ih : List.Pairwise (fun a₁ a₂ => ∀ (a : Fin k → Nat), Membership.mem (List.N …
        ⊢ List.Pairwise (fun a b => ∀ (a_1 : Fin k → Nat), Membership.mem (List.Nat.an …
      -/
      exact n_ih
      /-
        🎉 no goals
      -/


/-- `Multiset.Nat.antidiagonalTuple k n` is a multiset of `k`-tuples summing to `n` -/
def antidiagonalTuple (k n : ℕ) : Multiset (Fin k → ℕ) :=
  List.Nat.antidiagonalTuple k n


@[simp]
theorem antidiagonalTuple_zero_zero : antidiagonalTuple 0 0 = {![]} :=
  rfl


@[simp]
theorem antidiagonalTuple_zero_succ (n : ℕ) : antidiagonalTuple 0 n.succ = 0 :=
  rfl


theorem mem_antidiagonalTuple {n : ℕ} {k : ℕ} {x : Fin k → ℕ} :
    x ∈ antidiagonalTuple k n ↔ ∑ i, x i = n :=
  List.Nat.mem_antidiagonalTuple


theorem nodup_antidiagonalTuple (k n : ℕ) : (antidiagonalTuple k n).Nodup :=
  List.Nat.nodup_antidiagonalTuple _ _


theorem antidiagonalTuple_zero_right (k : ℕ) : antidiagonalTuple k 0 = {0} :=
  congr_arg _ (List.Nat.antidiagonalTuple_zero_right k)


@[simp]
theorem antidiagonalTuple_one (n : ℕ) : antidiagonalTuple 1 n = {![n]} :=
  congr_arg _ (List.Nat.antidiagonalTuple_one n)


theorem antidiagonalTuple_two (n : ℕ) :
    antidiagonalTuple 2 n = (antidiagonal n).map fun i => ![i.1, i.2] :=
  congr_arg _ (List.Nat.antidiagonalTuple_two n)


/-- `Finset.Nat.antidiagonalTuple k n` is a finset of `k`-tuples summing to `n` -/
def antidiagonalTuple (k n : ℕ) : Finset (Fin k → ℕ) :=
  ⟨Multiset.Nat.antidiagonalTuple k n, Multiset.Nat.nodup_antidiagonalTuple k n⟩


@[simp]
theorem antidiagonalTuple_zero_succ (n : ℕ) : antidiagonalTuple 0 n.succ = ∅ :=
  rfl


theorem antidiagonalTuple_zero_right (k : ℕ) : antidiagonalTuple k 0 = {0} :=
  Finset.eq_of_veq (Multiset.Nat.antidiagonalTuple_zero_right k)


@[simp]
theorem antidiagonalTuple_one (n : ℕ) : antidiagonalTuple 1 n = {![n]} :=
  Finset.eq_of_veq (Multiset.Nat.antidiagonalTuple_one n)


theorem antidiagonalTuple_two (n : ℕ) :
    antidiagonalTuple 2 n = (antidiagonal n).map (piFinTwoEquiv fun _ => ℕ).symm.toEmbedding :=
  Finset.eq_of_veq (Multiset.Nat.antidiagonalTuple_two n)


/-- The disjoint union of antidiagonal tuples `Σ n, antidiagonalTuple k n` is equivalent to the
`k`-tuple `Fin k → ℕ`. This is such an equivalence, obtained by mapping `(n, x)` to `x`.

This is the tuple version of `Finset.sigmaAntidiagonalEquivProd`. -/
@[simps]
def sigmaAntidiagonalTupleEquivTuple (k : ℕ) : (Σ n, antidiagonalTuple k n) ≃ (Fin k → ℕ) where
  toFun x := x.2
  invFun x := ⟨∑ i, x i, x, mem_antidiagonalTuple.mpr rfl⟩
  left_inv := fun ⟨_, _, h⟩ => Sigma.subtype_ext (mem_antidiagonalTuple.mp h) rfl
  right_inv _ := rfl



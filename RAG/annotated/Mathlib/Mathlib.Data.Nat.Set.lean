theorem zero_union_range_succ : {0} ∪ range succ = univ := by
  /-
    ⊢ Eq (Union.union (Singleton.singleton 0) (Set.range Nat.succ)) Set.univ
  -/
  ext n
  /-
    case h
    n : Nat
    ⊢ Iff (Membership.mem (Union.union (Singleton.singleton 0) (Set.range Nat.succ …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


@[simp]
protected theorem range_succ : range succ = { i | 0 < i } := by
  /-
    ⊢ Eq (Set.range Nat.succ) (setOf fun i => LT.lt 0 i)
  -/
                  /-
                    🎉 no goals
                  -/
  ext (_ | i) <;> simp [succ_pos, succ_ne_zero, Set.mem_setOf]
                  /-
                    🎉 no goals
                  -/


theorem range_of_succ (f : ℕ → α) : {f 0} ∪ range (f ∘ succ) = range f := by
  /-
    α : Type u_1
    f : Nat → α
    ⊢ Eq (Union.union (Singleton.singleton (f 0)) (Set.range (Function.comp f Nat. …
  -/
  rw [← image_singleton, range_comp, ← image_union, zero_union_range_succ, image_univ]
  /-
    🎉 no goals
  -/


theorem range_rec {α : Type*} (x : α) (f : ℕ → α → α) :
    (Set.range fun n => Nat.rec x f n : Set α) =
      {x} ∪ Set.range fun n => Nat.rec (f 0 x) (f ∘ succ) n := by
  /-
    α : Type u_2
    x : α
    f : Nat → α → α
    ⊢ Eq (Set.range fun n => Nat.rec x f n) (Union.union (Singleton.singleton x) ( …
  -/
  convert (range_of_succ (fun n => Nat.rec x f n : ℕ → α)).symm using 4
  /-
    case h.e'_3.h.e'_4.h.e'_3.h
    α : Type u_2
    x : α
    f : Nat → α → α
    x✝ : Nat
    ⊢ Eq (Nat.rec (f 0 x) (Function.comp f Nat.succ) x✝) (Function.comp (fun n =>  …
  -/
  dsimp
  /-
    case h.e'_3.h.e'_4.h.e'_3.h
    α : Type u_2
    x : α
    f : Nat → α → α
    x✝ : Nat
    ⊢ Eq (Nat.rec (f 0 x) (Function.comp f Nat.succ) x✝) (f x✝ (Nat.rec x f x✝))
  -/
  rename_i n
  induction n with
  | zero => rfl
  | succ n ihn => dsimp at ihn ⊢; rw [ihn]


theorem range_casesOn {α : Type*} (x : α) (f : ℕ → α) :
    (Set.range fun n => Nat.casesOn n x f : Set α) = {x} ∪ Set.range f :=
  (range_of_succ _).symm



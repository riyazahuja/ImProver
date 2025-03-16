private lemma Icc_neg_mono : Monotone fun n : ℕ ↦ Icc (-n : α) n := by
  /-
    α : Type u_1
    inst✝¹ : OrderedRing α
    inst✝ : LocallyFiniteOrder α
    ⊢ Monotone fun n => Finset.Icc (Neg.neg ↑n) ↑n
  -/
  refine fun m n hmn ↦ by apply Icc_subset_Icc <;> simpa using Nat.mono_cast hmn
  /-
    🎉 no goals
  -/


/-- Hollow box centered at `0 : α` going from `-n` to `n`. -/
def box : ℕ → Finset α := disjointed fun n ↦ Icc (-n : α) n


                                                        /-
                                                          α : Type u_1
                                                          inst✝² : OrderedRing α
                                                          inst✝¹ : LocallyFiniteOrder α
                                                          inst✝ : DecidableEq α
                                                          ⊢ Eq (Finset.box 0) (Singleton.singleton 0)
                                                        -/
@[simp] lemma box_zero : (box 0 : Finset α) = {0} := by simp [box]
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma box_succ_eq_sdiff (n : ℕ) :
    box (n + 1) = Icc (-n.succ : α) n.succ \ Icc (-n) n := Icc_neg_mono.disjointed_succ _


lemma disjoint_box_succ_prod (n : ℕ) : Disjoint (box (n + 1)) (Icc (-n : α) n) := by
  /-
    α : Type u_1
    inst✝² : OrderedRing α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    n : Nat
    ⊢ Disjoint (Finset.box (HAdd.hAdd n 1)) (Finset.Icc (Neg.neg ↑n) ↑n)
  -/
  rw [box_succ_eq_sdiff]; exact disjoint_sdiff_self_left
                          /-
                            🎉 no goals
                          -/


@[simp] lemma box_succ_union_prod (n : ℕ) :
    box (n + 1) ∪ Icc (-n : α) n = Icc (-n.succ : α) n.succ := Icc_neg_mono.disjointed_succ_sup _


lemma box_succ_disjUnion (n : ℕ) :
    (box (n + 1)).disjUnion (Icc (-n : α) n) (disjoint_box_succ_prod _) =
                                     /-
                                       α : Type u_1
                                       inst✝² : OrderedRing α
                                       inst✝¹ : LocallyFiniteOrder α
                                       inst✝ : DecidableEq α
                                       n : Nat
                                       ⊢ Eq ((Finset.box (HAdd.hAdd n 1)).disjUnion (Finset.Icc (Neg.neg ↑n) ↑n) ⋯) ( …
                                     -/
      Icc (-n.succ : α) n.succ := by rw [disjUnion_eq_union, box_succ_union_prod]
                                     /-
                                       🎉 no goals
                                     -/


                                                           /-
                                                             α : Type u_1
                                                             inst✝² : OrderedRing α
                                                             inst✝¹ : LocallyFiniteOrder α
                                                             n : Nat
                                                             inst✝ : DecidableEq α
                                                             ⊢ Iff (Membership.mem (Finset.box n) 0) (Eq n 0)
                                                           -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
@[simp] lemma zero_mem_box : (0 : α) ∈ box n ↔ n = 0 := by cases n <;> simp [box_succ_eq_sdiff]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma eq_zero_iff_eq_zero_of_mem_box  {x : α} (hx : x ∈ box n) : x = 0 ↔ n = 0 :=
                                           /-
                                             α : Type u_1
                                             inst✝² : OrderedRing α
                                             inst✝¹ : LocallyFiniteOrder α
                                             n : Nat
                                             inst✝ : DecidableEq α
                                             x : α
                                             hx : Membership.mem (Finset.box n) x
                                             hn : Eq n 0
                                             ⊢ Eq x 0
                                           -/
  ⟨zero_mem_box.mp ∘ (· ▸ hx), fun hn ↦ by rwa [hn, box_zero, mem_singleton] at hx⟩
                                           /-
                                             🎉 no goals
                                           -/


@[simp] lemma card_box_succ (n : ℕ) :
    #(box (n + 1) : Finset (α × β)) =
      #(Icc (-n.succ : α) n.succ) * #(Icc (-n.succ : β) n.succ) -
        #(Icc (-n : α) n) * #(Icc (-n : β) n) := by
  rw [box_succ_eq_sdiff, card_sdiff (Icc_neg_mono n.le_succ), Finset.card_Icc_prod,
    Finset.card_Icc_prod]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : OrderedRing α
    inst✝⁵ : OrderedRing β
    inst✝⁴ : LocallyFiniteOrder α
    inst✝³ : LocallyFiniteOrder β
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    n : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (Finset.Icc (Neg.neg ↑n.succ).1 (↑n.succ).1).card ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma card_box : ∀ {n}, n ≠ 0 → #(box n : Finset (ℤ × ℤ)) = 8 * n
  | n + 1, _ => by
    /-
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (Finset.box (HAdd.hAdd n 1)).card (HMul.hMul 8 (HAdd.hAdd n 1))
    -/
    simp_rw [Prod.card_box_succ, card_Icc, sub_neg_eq_add]
    /-
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑n.succ) 1) ↑n.succ).toNat ( …
    -/
    norm_cast
    /-
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HAdd.hAdd n.succ 1) n.succ) (HAdd.hAdd  …
    -/
    refine tsub_eq_of_eq_add ?_
    /-
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd n.succ 1) n.succ) (HAdd.hAdd (HAdd.hAdd  …
    -/
    zify
    /-
      n : Nat
      x✝ : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) (HAdd.hAdd (↑n) 1) …
    -/
    ring
    /-
      🎉 no goals
    -/


@[simp] lemma mem_box : ∀ {n}, x ∈ box n ↔ max x.1.natAbs x.2.natAbs = n
            /-
              x : Prod Int Int
              ⊢ Iff (Membership.mem (Finset.box 0) x) (Eq (Max.max x.1.natAbs x.2.natAbs) 0)
            -/
  | 0 => by simp [Prod.ext_iff]
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      x : Prod Int Int
      n : Nat
      ⊢ Iff (Membership.mem (Finset.box (HAdd.hAdd n 1)) x) (Eq (Max.max x.1.natAbs  …
    -/
    simp [box_succ_eq_sdiff, Prod.le_def]
    /-
      x : Prod Int Int
      n : Nat
      ⊢ Iff (And (And (And (LE.le (-1) (HAdd.hAdd x.1 ↑n)) (LE.le (-1) (HAdd.hAdd x. …
    -/
    omega
    /-
      🎉 no goals
    -/

-- TODO: Can this be generalised to locally finite archimedean ordered rings?

lemma existsUnique_mem_box (x : ℤ × ℤ) : ∃! n : ℕ, x ∈ box n := by
  /-
    x : Prod Int Int
    ⊢ ExistsUnique fun n => Membership.mem (Finset.box n) x
  -/
  use max x.1.natAbs x.2.natAbs; simp only [mem_box, and_self_iff, forall_eq']
                                 /-
                                   🎉 no goals
                                 -/



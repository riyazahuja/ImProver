/-- `m.sym2` is the multiset of all unordered pairs of elements from `m`, with multiplicity.
If `m` has no duplicates then neither does `m.sym2`. -/
protected def sym2 (m : Multiset α) : Multiset (Sym2 α) :=
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 m : Multiset α
                                                 x✝¹ x✝ : List α
                                                 h : HasEquiv.Equiv x✝¹ x✝
                                                 ⊢ Eq ((fun xs => ↑xs.sym2) x✝¹) ((fun xs => ↑xs.sym2) x✝)
                                               -/
  m.liftOn (fun xs => xs.sym2) fun _ _ h => by rw [coe_eq_coe]; exact h.sym2
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp] theorem sym2_coe (xs : List α) : (xs : Multiset α).sym2 = xs.sym2 := rfl


@[simp]
theorem sym2_eq_zero_iff {m : Multiset α} : m.sym2 = 0 ↔ m = 0 :=
                             /-
                               α : Type u_1
                               m : Multiset α
                               xs : List α
                               ⊢ Iff (Eq (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)) 0) (Eq (Quotient. …
                             -/
  m.inductionOn fun xs => by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem sym2_zero : (0 : Multiset α).sym2 = 0 := rfl


theorem sym2_cons (a : α) (m : Multiset α) :
    (m.cons a).sym2 = ((m.cons a).map <| fun b => s(a, b)) + m.sym2 :=
  m.inductionOn fun _ => rfl


theorem sym2_map (f : α → β) (m : Multiset α) :
    (m.map f).sym2 = m.sym2.map (Sym2.map f) :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               f : α → β
                               m : Multiset α
                               xs : List α
                               ⊢ Eq (Multiset.map f (Quotient.mk (List.isSetoid α) xs)).sym2 (Multiset.map (S …
                             -/
  m.inductionOn fun xs => by simp [List.sym2_map]
                             /-
                               🎉 no goals
                             -/


theorem mk_mem_sym2_iff {m : Multiset α} {a b : α} :
    s(a, b) ∈ m.sym2 ↔ a ∈ m ∧ b ∈ m :=
                             /-
                               α : Type u_1
                               m : Multiset α
                               a b : α
                               xs : List α
                               ⊢ Iff (Membership.mem (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)) (Sym2 …
                             -/
  m.inductionOn fun xs => by simp [List.mk_mem_sym2_iff]
                             /-
                               🎉 no goals
                             -/


theorem mem_sym2_iff {m : Multiset α} {z : Sym2 α} :
    z ∈ m.sym2 ↔ ∀ y ∈ z, y ∈ m :=
                             /-
                               α : Type u_1
                               m : Multiset α
                               z : Sym2 α
                               xs : List α
                               ⊢ Iff (Membership.mem (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)) z) (∀ …
                             -/
  m.inductionOn fun xs => by simp [List.mem_sym2_iff]
                             /-
                               🎉 no goals
                             -/


protected theorem Nodup.sym2 {m : Multiset α} (h : m.Nodup) : m.sym2.Nodup :=
  m.inductionOn (fun _ h => List.Nodup.sym2 h) h


open scoped List in
@[simp, mono]
theorem sym2_mono {m m' : Multiset α} (h : m ≤ m') : m.sym2 ≤ m'.sym2 := by
  /-
    α : Type u_1
    m m' : Multiset α
    h : LE.le m m'
    ⊢ LE.le m.sym2 m'.sym2
  -/
  refine Quotient.inductionOn₂ m m' (fun xs ys h => ?_) h
  /-
    α : Type u_1
    m m' : Multiset α
    h✝ : LE.le m m'
    xs ys : List α
    h : LE.le (Quotient.mk (List.isSetoid α) xs) (Quotient.mk (List.isSetoid α) ys)
    ⊢ LE.le (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)) (Multiset.sym2 (Quo …
  -/
  suffices xs <+~ ys from this.sym2
  /-
    α : Type u_1
    m m' : Multiset α
    h✝ : LE.le m m'
    xs ys : List α
    h : LE.le (Quotient.mk (List.isSetoid α) xs) (Quotient.mk (List.isSetoid α) ys)
    ⊢ xs.Subperm ys
  -/
  simpa only [quot_mk_to_coe, coe_le, sym2_coe] using h
  /-
    🎉 no goals
  -/


theorem monotone_sym2 : Monotone (Multiset.sym2 : Multiset α → _) := fun _ _ => sym2_mono


theorem card_sym2 {m : Multiset α} :
    Multiset.card m.sym2 = Nat.choose (Multiset.card m + 1) 2 := by
  /-
    α : Type u_1
    m : Multiset α
    ⊢ Eq m.sym2.card ((HAdd.hAdd m.card 1).choose 2)
  -/
  refine m.inductionOn fun xs => ?_
  /-
    α : Type u_1
    m : Multiset α
    xs : List α
    ⊢ Eq (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)).card ((HAdd.hAdd (Mult …
  -/
  simp [List.length_sym2]
  /-
    🎉 no goals
  -/


theorem dedup_sym2 [DecidableEq α] (m : Multiset α) : m.sym2.dedup = m.dedup.sym2 :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               m : Multiset α
                               xs : List α
                               ⊢ Eq (Multiset.sym2 (Quotient.mk (List.isSetoid α) xs)).dedup (Multiset.dedup  …
                             -/
  m.inductionOn fun xs => by simp [List.dedup_sym2]
                             /-
                               🎉 no goals
                             -/



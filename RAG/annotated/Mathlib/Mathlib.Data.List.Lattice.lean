@[symm]
theorem Disjoint.symm (d : Disjoint l₁ l₂) : Disjoint l₂ l₁ := fun _ i₂ i₁ => d i₁ i₂


theorem mem_union_left (h : a ∈ l₁) (l₂ : List α) : a ∈ l₁ ∪ l₂ :=
  mem_union_iff.2 (Or.inl h)


theorem mem_union_right (l₁ : List α) (h : a ∈ l₂) : a ∈ l₁ ∪ l₂ :=
  mem_union_iff.2 (Or.inr h)


theorem sublist_suffix_of_union : ∀ l₁ l₂ : List α, ∃ t, t <+ l₁ ∧ t ++ l₂ = l₁ ∪ l₂
                     /-
                       α : Type u_1
                       inst✝ : DecidableEq α
                       x✝ : List α
                       ⊢ List.nil.Sublist List.nil
                     -/
  | [], _ => ⟨[], by rfl, rfl⟩
                     /-
                       🎉 no goals
                     -/
  | a :: l₁, l₂ =>
    let ⟨t, s, e⟩ := sublist_suffix_of_union l₁ l₂
    if h : a ∈ l₁ ∪ l₂ then
      ⟨t, sublist_cons_of_sublist _ s, by
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          l₁ l₂ t : List α
          s : t.Sublist l₁
          e : Eq (HAppend.hAppend t l₂) (Union.union l₁ l₂)
          h : Membership.mem (Union.union l₁ l₂) a
          ⊢ Eq (HAppend.hAppend t l₂) (Union.union (List.cons a l₁) l₂)
        -/
        simp only [e, cons_union, insert_of_mem h]⟩
        /-
          🎉 no goals
        -/
    else
      ⟨a :: t, s.cons_cons _, by
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          l₁ l₂ t : List α
          s : t.Sublist l₁
          e : Eq (HAppend.hAppend t l₂) (Union.union l₁ l₂)
          h : Not (Membership.mem (Union.union l₁ l₂) a)
          ⊢ Eq (HAppend.hAppend (List.cons a t) l₂) (Union.union (List.cons a l₁) l₂)
        -/
        simp only [cons_append, cons_union, e, insert_of_not_mem h]⟩
        /-
          🎉 no goals
        -/


theorem suffix_union_right (l₁ l₂ : List α) : l₂ <:+ l₁ ∪ l₂ :=
  (sublist_suffix_of_union l₁ l₂).imp fun _ => And.right


theorem union_sublist_append (l₁ l₂ : List α) : l₁ ∪ l₂ <+ l₁ ++ l₂ :=
  let ⟨_, s, e⟩ := sublist_suffix_of_union l₁ l₂
  e ▸ (append_sublist_append_right _).2 s


theorem forall_mem_union : (∀ x ∈ l₁ ∪ l₂, p x) ↔ (∀ x ∈ l₁, p x) ∧ ∀ x ∈ l₂, p x := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    p : α → Prop
    inst✝ : DecidableEq α
    ⊢ Iff (∀ (x : α), Membership.mem (Union.union l₁ l₂) x → p x) (And (∀ (x : α), …
  -/
  simp only [mem_union_iff, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem forall_mem_of_forall_mem_union_left (h : ∀ x ∈ l₁ ∪ l₂, p x) : ∀ x ∈ l₁, p x :=
  (forall_mem_union.1 h).1


theorem forall_mem_of_forall_mem_union_right (h : ∀ x ∈ l₁ ∪ l₂, p x) : ∀ x ∈ l₂, p x :=
  (forall_mem_union.1 h).2


theorem Subset.union_eq_right {xs ys : List α} (h : xs ⊆ ys) : xs ∪ ys = ys := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    rw [cons_union, insert_of_mem <| mem_union_right _ <| h <| mem_cons_self _ _,
      ih <| subset_of_cons_subset h]


@[simp]
theorem inter_nil (l : List α) : [] ∩ l = [] :=
  rfl


@[simp]
theorem inter_cons_of_mem (l₁ : List α) (h : a ∈ l₂) : (a :: l₁) ∩ l₂ = a :: l₁ ∩ l₂ := by
  /-
    α : Type u_1
    l₂ : List α
    a : α
    inst✝ : DecidableEq α
    l₁ : List α
    h : Membership.mem l₂ a
    ⊢ Eq (Inter.inter (List.cons a l₁) l₂) (List.cons a (Inter.inter l₁ l₂))
  -/
  simp [Inter.inter, List.inter, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_cons_of_not_mem (l₁ : List α) (h : a ∉ l₂) : (a :: l₁) ∩ l₂ = l₁ ∩ l₂ := by
  /-
    α : Type u_1
    l₂ : List α
    a : α
    inst✝ : DecidableEq α
    l₁ : List α
    h : Not (Membership.mem l₂ a)
    ⊢ Eq (Inter.inter (List.cons a l₁) l₂) (Inter.inter l₁ l₂)
  -/
  simp [Inter.inter, List.inter, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_nil' (l : List α) : l ∩ [] = [] := by
  induction l with
  | nil => rfl
  | cons x xs ih => by_cases x ∈ xs <;> simp [ih]


theorem mem_of_mem_inter_left : a ∈ l₁ ∩ l₂ → a ∈ l₁ :=
  mem_of_mem_filter


                                                                /-
                                                                  α : Type u_1
                                                                  l₁ l₂ : List α
                                                                  a : α
                                                                  inst✝ : DecidableEq α
                                                                  h : Membership.mem (Inter.inter l₁ l₂) a
                                                                  ⊢ Membership.mem l₂ a
                                                                -/
theorem mem_of_mem_inter_right (h : a ∈ l₁ ∩ l₂) : a ∈ l₂ := by simpa using of_mem_filter h
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem mem_inter_of_mem_of_mem (h₁ : a ∈ l₁) (h₂ : a ∈ l₂) : a ∈ l₁ ∩ l₂ :=
                             /-
                               α : Type u_1
                               l₁ l₂ : List α
                               a : α
                               inst✝ : DecidableEq α
                               h₁ : Membership.mem l₁ a
                               h₂ : Membership.mem l₂ a
                               ⊢ Eq (List.elem a l₂) Bool.true
                             -/
  mem_filter_of_mem h₁ <| by simpa using h₂
                             /-
                               🎉 no goals
                             -/


theorem inter_subset_left {l₁ l₂ : List α} : l₁ ∩ l₂ ⊆ l₁ :=
  filter_subset' _


theorem inter_subset_right {l₁ l₂ : List α} : l₁ ∩ l₂ ⊆ l₂ := fun _ => mem_of_mem_inter_right


theorem subset_inter {l l₁ l₂ : List α} (h₁ : l ⊆ l₁) (h₂ : l ⊆ l₂) : l ⊆ l₁ ∩ l₂ := fun _ h =>
  mem_inter_iff.2 ⟨h₁ h, h₂ h⟩


theorem inter_eq_nil_iff_disjoint : l₁ ∩ l₂ = [] ↔ Disjoint l₁ l₂ := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    ⊢ Iff (Eq (Inter.inter l₁ l₂) List.nil) (l₁.Disjoint l₂)
  -/
  simp only [eq_nil_iff_forall_not_mem, mem_inter_iff, not_and]
  /-
    α : Type u_1
    l₁ l₂ : List α
    inst✝ : DecidableEq α
    ⊢ Iff (∀ (a : α), Membership.mem l₁ a → Not (Membership.mem l₂ a)) (l₁.Disjoin …
  -/
  rfl
  /-
    🎉 no goals
  -/


alias ⟨_, Disjoint.inter_eq_nil⟩ := inter_eq_nil_iff_disjoint


theorem forall_mem_inter_of_forall_left (h : ∀ x ∈ l₁, p x) (l₂ : List α) :
    ∀ x, x ∈ l₁ ∩ l₂ → p x :=
  BAll.imp_left (fun _ => mem_of_mem_inter_left) h


theorem forall_mem_inter_of_forall_right (l₁ : List α) (h : ∀ x ∈ l₂, p x) :
    ∀ x, x ∈ l₁ ∩ l₂ → p x :=
  BAll.imp_left (fun _ => mem_of_mem_inter_right) h


@[simp]
theorem inter_reverse {xs ys : List α} : xs.inter ys.reverse = xs.inter ys := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    ⊢ Eq (xs.inter ys.reverse) (xs.inter ys)
  -/
  simp only [List.inter, elem_eq_mem, mem_reverse]
  /-
    🎉 no goals
  -/


theorem Subset.inter_eq_left {xs ys : List α} (h : xs ⊆ ys) : xs ∩ ys = xs :=
  List.filter_eq_self.mpr fun _ ha => elem_eq_true_of_mem (h ha)


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : DecidableEq α
                                                               l : List α
                                                               ⊢ Eq (List.nil.bagInter l) List.nil
                                                             -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
theorem nil_bagInter (l : List α) : [].bagInter l = [] := by cases l <;> rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               inst✝ : DecidableEq α
                                                               l : List α
                                                               ⊢ Eq (l.bagInter List.nil) List.nil
                                                             -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
theorem bagInter_nil (l : List α) : l.bagInter [] = [] := by cases l <;> rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem cons_bagInter_of_pos (l₁ : List α) (h : a ∈ l₂) :
    (a :: l₁).bagInter l₂ = a :: l₁.bagInter (l₂.erase a) := by
  /-
    α : Type u_1
    l₂ : List α
    a : α
    inst✝ : DecidableEq α
    l₁ : List α
    h : Membership.mem l₂ a
    ⊢ Eq ((List.cons a l₁).bagInter l₂) (List.cons a (l₁.bagInter (l₂.erase a)))
  -/
  cases l₂
    /-
      case nil
      α : Type u_1
      a : α
      inst✝ : DecidableEq α
      l₁ : List α
      h : Membership.mem List.nil a
      ⊢ Eq ((List.cons a l₁).bagInter List.nil) (List.cons a (l₁.bagInter (List.nil. …
    -/
  · exact if_pos h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      a : α
      inst✝ : DecidableEq α
      l₁ : List α
      head✝ : α
      tail✝ : List α
      h : Membership.mem (List.cons head✝ tail✝) a
      ⊢ Eq ((List.cons a l₁).bagInter (List.cons head✝ tail✝)) (List.cons a (l₁.bagI …
    -/
  · simp only [List.bagInter, if_pos (elem_eq_true_of_mem h)]
    /-
      🎉 no goals
    -/


@[simp]
theorem cons_bagInter_of_neg (l₁ : List α) (h : a ∉ l₂) :
    (a :: l₁).bagInter l₂ = l₁.bagInter l₂ := by
  /-
    α : Type u_1
    l₂ : List α
    a : α
    inst✝ : DecidableEq α
    l₁ : List α
    h : Not (Membership.mem l₂ a)
    ⊢ Eq ((List.cons a l₁).bagInter l₂) (l₁.bagInter l₂)
  -/
  cases l₂; · simp only [bagInter_nil]
              /-
                🎉 no goals
              -/
  /-
    case cons
    α : Type u_1
    a : α
    inst✝ : DecidableEq α
    l₁ : List α
    head✝ : α
    tail✝ : List α
    h : Not (Membership.mem (List.cons head✝ tail✝) a)
    ⊢ Eq ((List.cons a l₁).bagInter (List.cons head✝ tail✝)) (l₁.bagInter (List.co …
  -/
  simp only [erase_of_not_mem h, List.bagInter, if_neg (mt mem_of_elem_eq_true h)]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_bagInter {a : α} : ∀ {l₁ l₂ : List α}, a ∈ l₁.bagInter l₂ ↔ a ∈ l₁ ∧ a ∈ l₂
                 /-
                   α : Type u_1
                   inst✝ : DecidableEq α
                   a : α
                   l₂ : List α
                   ⊢ Iff (Membership.mem (List.nil.bagInter l₂) a) (And (Membership.mem List.nil  …
                 -/
  | [], l₂ => by simp only [nil_bagInter, not_mem_nil, false_and]
                 /-
                   🎉 no goals
                 -/
  | b :: l₁, l₂ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      l₁ l₂ : List α
      ⊢ Iff (Membership.mem ((List.cons b l₁).bagInter l₂) a) (And (Membership.mem ( …
    -/
    by_cases h : b ∈ l₂
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Membership.mem l₂ b
        ⊢ Iff (Membership.mem ((List.cons b l₁).bagInter l₂) a) (And (Membership.mem ( …
      -/
    · rw [cons_bagInter_of_pos _ h, mem_cons, mem_cons, mem_bagInter]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Membership.mem l₂ b
        ⊢ Iff (Or (Eq a b) (And (Membership.mem l₁ a) (Membership.mem (l₂.erase b) a)) …
      -/
      by_cases ba : a = b
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          h : Membership.mem l₂ b
          ba : Eq a b
          ⊢ Iff (Or (Eq a b) (And (Membership.mem l₁ a) (Membership.mem (l₂.erase b) a)) …
        -/
      · simp only [ba, h, eq_self_iff_true, true_or, true_and]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          h : Membership.mem l₂ b
          ba : Not (Eq a b)
          ⊢ Iff (Or (Eq a b) (And (Membership.mem l₁ a) (Membership.mem (l₂.erase b) a)) …
        -/
      · simp only [mem_erase_of_ne ba, ba, false_or]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ Iff (Membership.mem ((List.cons b l₁).bagInter l₂) a) (And (Membership.mem ( …
      -/
    · rw [cons_bagInter_of_neg _ h, mem_bagInter, mem_cons, or_and_right]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ Iff (And (Membership.mem l₁ a) (Membership.mem l₂ a)) (Or (And (Eq a b) (Mem …
      -/
      symm
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ Iff (Or (And (Eq a b) (Membership.mem l₂ a)) (And (Membership.mem l₁ a) (Mem …
      -/
      apply or_iff_right_of_imp
      /-
        case neg.ha
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ And (Eq a b) (Membership.mem l₂ a) → And (Membership.mem l₁ a) (Membership.m …
      -/
      rintro ⟨rfl, h'⟩
      /-
        case neg.ha.intro
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l₁ l₂ : List α
        h' : Membership.mem l₂ a
        h : Not (Membership.mem l₂ a)
        ⊢ And (Membership.mem l₁ a) (Membership.mem l₂ a)
      -/
      exact h.elim h'
      /-
        🎉 no goals
      -/


@[simp]
theorem count_bagInter {a : α} :
    ∀ {l₁ l₂ : List α}, count a (l₁.bagInter l₂) = min (count a l₁) (count a l₂)
                 /-
                   α : Type u_1
                   inst✝ : DecidableEq α
                   a : α
                   l₂ : List α
                   ⊢ Eq (List.count a (List.nil.bagInter l₂)) (Min.min (List.count a List.nil) (L …
                 -/
  | [], l₂ => by simp
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   inst✝ : DecidableEq α
                   a : α
                   l₁ : List α
                   ⊢ Eq (List.count a (l₁.bagInter List.nil)) (Min.min (List.count a l₁) (List.co …
                 -/
  | l₁, [] => by simp
                 /-
                   🎉 no goals
                 -/
  | b :: l₁, l₂ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      l₁ l₂ : List α
      ⊢ Eq (List.count a ((List.cons b l₁).bagInter l₂)) (Min.min (List.count a (Lis …
    -/
    by_cases hb : b ∈ l₂
    · rw [cons_bagInter_of_pos _ hb, count_cons, count_cons, count_bagInter, count_erase,
        ← Nat.add_min_add_right]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        hb : Membership.mem l₂ b
        ⊢ Eq (Min.min (HAdd.hAdd (List.count a l₁) (ite (Eq (BEq.beq b a) Bool.true) 1 …
      -/
      by_cases ba : b = a
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Membership.mem l₂ b
          ba : Eq b a
          ⊢ Eq (Min.min (HAdd.hAdd (List.count a l₁) (ite (Eq (BEq.beq b a) Bool.true) 1 …
        -/
      · simp only [beq_iff_eq]
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Membership.mem l₂ b
          ba : Eq b a
          ⊢ Eq (Min.min (HAdd.hAdd (List.count a l₁) (ite (Eq b a) 1 0)) (HAdd.hAdd (HSu …
        -/
        rw [if_pos ba, Nat.sub_add_cancel]
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Membership.mem l₂ b
          ba : Eq b a
          ⊢ LE.le 1 (List.count a l₂)
        -/
        rwa [succ_le_iff, count_pos_iff, ← ba]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Membership.mem l₂ b
          ba : Not (Eq b a)
          ⊢ Eq (Min.min (HAdd.hAdd (List.count a l₁) (ite (Eq (BEq.beq b a) Bool.true) 1 …
        -/
      · simp only [beq_iff_eq]
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Membership.mem l₂ b
          ba : Not (Eq b a)
          ⊢ Eq (Min.min (HAdd.hAdd (List.count a l₁) (ite (Eq b a) 1 0)) (HAdd.hAdd (HSu …
        -/
        rw [if_neg ba, Nat.sub_zero, Nat.add_zero, Nat.add_zero]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        hb : Not (Membership.mem l₂ b)
        ⊢ Eq (List.count a ((List.cons b l₁).bagInter l₂)) (Min.min (List.count a (Lis …
      -/
    · rw [cons_bagInter_of_neg _ hb, count_bagInter]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a b : α
        l₁ l₂ : List α
        hb : Not (Membership.mem l₂ b)
        ⊢ Eq (Min.min (List.count a l₁) (List.count a l₂)) (Min.min (List.count a (Lis …
      -/
      by_cases ab : a = b
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Not (Membership.mem l₂ b)
          ab : Eq a b
          ⊢ Eq (Min.min (List.count a l₁) (List.count a l₂)) (Min.min (List.count a (Lis …
        -/
      · rw [← ab] at hb
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Not (Membership.mem l₂ a)
          ab : Eq a b
          ⊢ Eq (Min.min (List.count a l₁) (List.count a l₂)) (Min.min (List.count a (Lis …
        -/
        rw [count_eq_zero.2 hb, Nat.min_zero, Nat.min_zero]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          a b : α
          l₁ l₂ : List α
          hb : Not (Membership.mem l₂ b)
          ab : Not (Eq a b)
          ⊢ Eq (Min.min (List.count a l₁) (List.count a l₂)) (Min.min (List.count a (Lis …
        -/
      · rw [count_cons_of_ne ab]
        /-
          🎉 no goals
        -/


theorem bagInter_sublist_left : ∀ l₁ l₂ : List α, l₁.bagInter l₂ <+ l₁
                 /-
                   α : Type u_1
                   inst✝ : DecidableEq α
                   l₂ : List α
                   ⊢ (List.nil.bagInter l₂).Sublist List.nil
                 -/
  | [], l₂ => by simp
                 /-
                   🎉 no goals
                 -/
  | b :: l₁, l₂ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      b : α
      l₁ l₂ : List α
      ⊢ ((List.cons b l₁).bagInter l₂).Sublist (List.cons b l₁)
    -/
    by_cases h : b ∈ l₂ <;> simp only [h, cons_bagInter_of_pos, cons_bagInter_of_neg, not_false_iff]
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        l₁ l₂ : List α
        h : Membership.mem l₂ b
        ⊢ (List.cons b (l₁.bagInter (l₂.erase b))).Sublist (List.cons b l₁)
      -/
    · exact (bagInter_sublist_left _ _).cons_cons _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ (l₁.bagInter l₂).Sublist (List.cons b l₁)
      -/
    · apply sublist_cons_of_sublist
      /-
        case neg.h
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ (l₁.bagInter l₂).Sublist l₁
      -/
      apply bagInter_sublist_left
      /-
        🎉 no goals
      -/


theorem bagInter_nil_iff_inter_nil : ∀ l₁ l₂ : List α, l₁.bagInter l₂ = [] ↔ l₁ ∩ l₂ = []
                 /-
                   α : Type u_1
                   inst✝ : DecidableEq α
                   l₂ : List α
                   ⊢ Iff (Eq (List.nil.bagInter l₂) List.nil) (Eq (Inter.inter List.nil l₂) List. …
                 -/
  | [], l₂ => by simp
                 /-
                   🎉 no goals
                 -/
  | b :: l₁, l₂ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      b : α
      l₁ l₂ : List α
      ⊢ Iff (Eq ((List.cons b l₁).bagInter l₂) List.nil) (Eq (Inter.inter (List.cons …
    -/
    by_cases h : b ∈ l₂
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        l₁ l₂ : List α
        h : Membership.mem l₂ b
        ⊢ Iff (Eq ((List.cons b l₁).bagInter l₂) List.nil) (Eq (Inter.inter (List.cons …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        l₁ l₂ : List α
        h : Not (Membership.mem l₂ b)
        ⊢ Iff (Eq ((List.cons b l₁).bagInter l₂) List.nil) (Eq (Inter.inter (List.cons …
      -/
    · simpa [h] using bagInter_nil_iff_inter_nil l₁ l₂
      /-
        🎉 no goals
      -/



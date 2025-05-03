/-- Disjoint sum of finsets. -/
def disjSum : Finset (α ⊕ β) :=
  ⟨s.1.disjSum t.1, s.2.disjSum t.2⟩


@[simp]
theorem val_disjSum : (s.disjSum t).1 = s.1.disjSum t.1 :=
  rfl


@[simp]
theorem empty_disjSum : (∅ : Finset α).disjSum t = t.map Embedding.inr :=
  val_inj.1 <| Multiset.zero_disjSum _


@[simp]
theorem disjSum_empty : s.disjSum (∅ : Finset β) = s.map Embedding.inl :=
  val_inj.1 <| Multiset.disjSum_zero _


@[simp]
theorem card_disjSum : (s.disjSum t).card = s.card + t.card :=
  Multiset.card_disjSum _ _


theorem disjoint_map_inl_map_inr : Disjoint (s.map Embedding.inl) (t.map Embedding.inr) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    ⊢ Disjoint (Finset.map Function.Embedding.inl s) (Finset.map Function.Embeddin …
  -/
  simp_rw [disjoint_left, mem_map]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    ⊢ ∀ ⦃a : Sum α β⦄, (Exists fun a_1 => And (Membership.mem s a_1) (Eq (Function …
  -/
  rintro x ⟨a, _, rfl⟩ ⟨b, _, ⟨⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem map_inl_disjUnion_map_inr :
    (s.map Embedding.inl).disjUnion (t.map Embedding.inr) (disjoint_map_inl_map_inr _ _) =
      s.disjSum t :=
  rfl


theorem mem_disjSum : x ∈ s.disjSum t ↔ (∃ a, a ∈ s ∧ inl a = x) ∨ ∃ b, b ∈ t ∧ inr b = x :=
  Multiset.mem_disjSum


@[simp]
theorem inl_mem_disjSum : inl a ∈ s.disjSum t ↔ a ∈ s :=
  Multiset.inl_mem_disjSum


@[simp]
theorem inr_mem_disjSum : inr b ∈ s.disjSum t ↔ b ∈ t :=
  Multiset.inr_mem_disjSum


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   s : Finset α
                                                                   t : Finset β
                                                                   ⊢ Iff (Eq (s.disjSum t) EmptyCollection.emptyCollection) (And (Eq s EmptyColle …
                                                                 -/
theorem disjSum_eq_empty : s.disjSum t = ∅ ↔ s = ∅ ∧ t = ∅ := by simp [Finset.ext_iff]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem disjSum_mono (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) : s₁.disjSum t₁ ⊆ s₂.disjSum t₂ :=
  val_le_iff.1 <| Multiset.disjSum_mono (val_le_iff.2 hs) (val_le_iff.2 ht)


theorem disjSum_mono_left (t : Finset β) : Monotone fun s : Finset α => s.disjSum t :=
  fun _ _ hs => disjSum_mono hs Subset.rfl


theorem disjSum_mono_right (s : Finset α) : Monotone (s.disjSum : Finset β → Finset (α ⊕ β)) :=
  fun _ _ => disjSum_mono Subset.rfl


theorem disjSum_ssubset_disjSum_of_ssubset_of_subset (hs : s₁ ⊂ s₂) (ht : t₁ ⊆ t₂) :
    s₁.disjSum t₁ ⊂ s₂.disjSum t₂ :=
  val_lt_iff.1 <| disjSum_lt_disjSum_of_lt_of_le (val_lt_iff.2 hs) (val_le_iff.2 ht)


theorem disjSum_ssubset_disjSum_of_subset_of_ssubset (hs : s₁ ⊆ s₂) (ht : t₁ ⊂ t₂) :
    s₁.disjSum t₁ ⊂ s₂.disjSum t₂ :=
  val_lt_iff.1 <| disjSum_lt_disjSum_of_le_of_lt (val_le_iff.2 hs) (val_lt_iff.2 ht)


theorem disjSum_strictMono_left (t : Finset β) : StrictMono fun s : Finset α => s.disjSum t :=
  fun _ _ hs => disjSum_ssubset_disjSum_of_ssubset_of_subset hs Subset.rfl


theorem disj_sum_strictMono_right (s : Finset α) :
    StrictMono (s.disjSum : Finset β → Finset (α ⊕ β)) := fun _ _ =>
  disjSum_ssubset_disjSum_of_subset_of_ssubset Subset.rfl


@[simp] lemma disjSum_inj {α β : Type*} {s₁ s₂ : Finset α} {t₁ t₂ : Finset β} :
    s₁.disjSum t₁ = s₂.disjSum t₂ ↔ s₁ = s₂ ∧ t₁ = t₂ := by
  /-
    α : Type u_3
    β : Type u_4
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    ⊢ Iff (Eq (s₁.disjSum t₁) (s₂.disjSum t₂)) (And (Eq s₁ s₂) (Eq t₁ t₂))
  -/
  simp [Finset.ext_iff]
  /-
    🎉 no goals
  -/


lemma Injective2_disjSum {α β : Type*} : Function.Injective2 (@disjSum α β) :=
                    /-
                      α : Type u_3
                      β : Type u_4
                      x✝³ x✝² : Finset α
                      x✝¹ x✝ : Finset β
                      ⊢ Eq (x✝³.disjSum x✝¹) (x✝².disjSum x✝) → And (Eq x✝³ x✝²) (Eq x✝¹ x✝)
                    -/
  fun _ _ _ _ => by simp [Finset.ext_iff]
                    /-
                      🎉 no goals
                    -/


/--
Given a finset of elements `α ⊕ β`, extract all the elements of the form `α`. This
forms a quasi-inverse to `disjSum`, in that it recovers its left input.

See also `List.partitionMap`.
-/
def toLeft (s : Finset (α ⊕ β)) : Finset α :=
  s.disjiUnion (Sum.elim singleton (fun _ => ∅)) <| by
    /-
      α : Type u_1
      β : Type u_2
      s✝ : Finset α
      t : Finset β
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      a : α
      b : β
      x : Sum α β
      s : Finset (Sum α β)
      ⊢ (↑s).PairwiseDisjoint (Sum.elim Singleton.singleton fun x => EmptyCollection …
    -/
    simp [Set.PairwiseDisjoint, Set.Pairwise, Function.onFun, eq_comm]
    /-
      🎉 no goals
    -/


/--
Given a finset of elements `α ⊕ β`, extract all the elements of the form `β`. This
forms a quasi-inverse to `disjSum`, in that it recovers its right input.

See also `List.partitionMap`.
-/
def toRight (s : Finset (α ⊕ β)) : Finset β :=
  s.disjiUnion (Sum.elim (fun _ => ∅) singleton) <| by
    /-
      α : Type u_1
      β : Type u_2
      s✝ : Finset α
      t : Finset β
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      a : α
      b : β
      x : Sum α β
      s : Finset (Sum α β)
      ⊢ (↑s).PairwiseDisjoint (Sum.elim (fun x => EmptyCollection.emptyCollection) S …
    -/
    simp [Set.PairwiseDisjoint, Set.Pairwise, Function.onFun, eq_comm]
    /-
      🎉 no goals
    -/


@[simp] lemma mem_toLeft {x : α} : x ∈ u.toLeft ↔ inl x ∈ u := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    x : α
    ⊢ Iff (Membership.mem u.toLeft x) (Membership.mem u (Sum.inl x))
  -/
  simp [toLeft]
  /-
    🎉 no goals
  -/


@[simp] lemma mem_toRight {x : β} : x ∈ u.toRight ↔ inr x ∈ u := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    x : β
    ⊢ Iff (Membership.mem u.toRight x) (Membership.mem u (Sum.inr x))
  -/
  simp [toRight]
  /-
    🎉 no goals
  -/


@[gcongr]
lemma toLeft_subset_toLeft : u ⊆ v → u.toLeft ⊆ v.toLeft :=
                /-
                  α : Type u_1
                  β : Type u_2
                  u v : Finset (Sum α β)
                  h : HasSubset.Subset u v
                  x✝ : α
                  ⊢ Membership.mem u.toLeft x✝ → Membership.mem v.toLeft x✝
                -/
  fun h _ => by simpa only [mem_toLeft] using @h _
                /-
                  🎉 no goals
                -/


@[gcongr]
lemma toRight_subset_toRight : u ⊆ v → u.toRight ⊆ v.toRight :=
                /-
                  α : Type u_1
                  β : Type u_2
                  u v : Finset (Sum α β)
                  h : HasSubset.Subset u v
                  x✝ : β
                  ⊢ Membership.mem u.toRight x✝ → Membership.mem v.toRight x✝
                -/
  fun h _ => by simpa only [mem_toRight] using @h _
                /-
                  🎉 no goals
                -/


lemma toLeft_monotone : Monotone (@toLeft α β) := fun _ _ => toLeft_subset_toLeft

lemma toRight_monotone : Monotone (@toRight α β) := fun _ _ => toRight_subset_toRight


lemma toLeft_disjSum_toRight : u.toLeft.disjSum u.toRight = u := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    ⊢ Eq (u.toLeft.disjSum u.toRight) u
  -/
                  /-
                    🎉 no goals
                  -/
  ext (x | x) <;> simp
                  /-
                    🎉 no goals
                  -/


lemma card_toLeft_add_card_toRight : u.toLeft.card + u.toRight.card = u.card := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    ⊢ Eq (HAdd.hAdd u.toLeft.card u.toRight.card) u.card
  -/
  rw [← card_disjSum, toLeft_disjSum_toRight]
  /-
    🎉 no goals
  -/


lemma card_toLeft_le : u.toLeft.card ≤ u.card :=
  (Nat.le_add_right _ _).trans_eq card_toLeft_add_card_toRight


lemma card_toRight_le : u.toRight.card ≤ u.card :=
  (Nat.le_add_left _ _).trans_eq card_toLeft_add_card_toRight


                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                s : Finset α
                                                                t : Finset β
                                                                ⊢ Eq (s.disjSum t).toLeft s
                                                              -/
@[simp] lemma toLeft_disjSum : (s.disjSum t).toLeft = s := by ext x; simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  s : Finset α
                                                                  t : Finset β
                                                                  ⊢ Eq (s.disjSum t).toRight t
                                                                -/
@[simp] lemma toRight_disjSum : (s.disjSum t).toRight = t := by ext x; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma disjSum_eq_iff : s.disjSum t = u ↔ s = u.toLeft ∧ t = u.toRight :=
               /-
                 α : Type u_1
                 β : Type u_2
                 s : Finset α
                 t : Finset β
                 u : Finset (Sum α β)
                 h : Eq (s.disjSum t) u
                 ⊢ And (Eq s u.toLeft) (Eq t u.toRight)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simp [← h], fun h => by simp [h, toLeft_disjSum_toRight]⟩
                                       /-
                                         🎉 no goals
                                       -/


lemma eq_disjSum_iff : u = s.disjSum t ↔ u.toLeft = s ∧ u.toRight = t :=
               /-
                 α : Type u_1
                 β : Type u_2
                 s : Finset α
                 t : Finset β
                 u : Finset (Sum α β)
                 h : Eq u (s.disjSum t)
                 ⊢ And (Eq u.toLeft s) (Eq u.toRight t)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simp [h], fun h => by simp [← h, toLeft_disjSum_toRight]⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp] lemma toLeft_map_sumComm : (u.map (Equiv.sumComm _ _).toEmbedding).toLeft = u.toRight := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    ⊢ Eq (Finset.map (Equiv.sumComm α β).toEmbedding u).toLeft u.toRight
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


@[simp] lemma toRight_map_sumComm : (u.map (Equiv.sumComm _ _).toEmbedding).toRight = u.toLeft := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    ⊢ Eq (Finset.map (Equiv.sumComm α β).toEmbedding u).toRight u.toLeft
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


@[simp] lemma toLeft_cons_inl (ha) :
                                                     /-
                                                       α : Type u_1
                                                       β : Type u_2
                                                       s : Finset α
                                                       t : Finset β
                                                       s₁ s₂ : Finset α
                                                       t₁ t₂ : Finset β
                                                       a : α
                                                       b : β
                                                       x : Sum α β
                                                       u v : Finset (Sum α β)
                                                       ha : Not (Membership.mem u (Sum.inl a))
                                                       ⊢ Not (Membership.mem u.toLeft a)
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    (cons (inl a) u ha).toLeft = cons a u.toLeft (by simpa) := by ext y; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

@[simp] lemma toLeft_cons_inr (hb) :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  b : β
                                                  u : Finset (Sum α β)
                                                  hb : Not (Membership.mem u (Sum.inr b))
                                                  ⊢ Eq (Finset.cons (Sum.inr b) u hb).toLeft u.toLeft
                                                -/
    (cons (inr b) u hb).toLeft = u.toLeft := by ext y; simp
                                                       /-
                                                         🎉 no goals
                                                       -/

@[simp] lemma toRight_cons_inl (ha) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    a : α
                                                    u : Finset (Sum α β)
                                                    ha : Not (Membership.mem u (Sum.inl a))
                                                    ⊢ Eq (Finset.cons (Sum.inl a) u ha).toRight u.toRight
                                                  -/
    (cons (inl a) u ha).toRight = u.toRight := by ext y; simp
                                                         /-
                                                           🎉 no goals
                                                         -/

@[simp] lemma toRight_cons_inr (hb) :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         s : Finset α
                                                         t : Finset β
                                                         s₁ s₂ : Finset α
                                                         t₁ t₂ : Finset β
                                                         a : α
                                                         b : β
                                                         x : Sum α β
                                                         u v : Finset (Sum α β)
                                                         hb : Not (Membership.mem u (Sum.inr b))
                                                         ⊢ Not (Membership.mem u.toRight b)
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    (cons (inr b) u hb).toRight = cons b u.toRight (by simpa) := by ext y; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma toLeft_image_swap : (u.image Sum.swap).toLeft = u.toRight := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (Finset.image Sum.swap u).toLeft u.toRight
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


lemma toRight_image_swap : (u.image Sum.swap).toRight = u.toLeft := by
  /-
    α : Type u_1
    β : Type u_2
    u : Finset (Sum α β)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (Finset.image Sum.swap u).toRight u.toLeft
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


                                                                                      /-
                                                                                        α : Type u_1
                                                                                        β : Type u_2
                                                                                        a : α
                                                                                        u : Finset (Sum α β)
                                                                                        inst✝¹ : DecidableEq α
                                                                                        inst✝ : DecidableEq β
                                                                                        ⊢ Eq (Insert.insert (Sum.inl a) u).toLeft (Insert.insert a u.toLeft)
                                                                                      -/
@[simp] lemma toLeft_insert_inl : (insert (inl a) u).toLeft = insert a u.toLeft := by ext y; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/

                                                                             /-
                                                                               α : Type u_1
                                                                               β : Type u_2
                                                                               b : β
                                                                               u : Finset (Sum α β)
                                                                               inst✝¹ : DecidableEq α
                                                                               inst✝ : DecidableEq β
                                                                               ⊢ Eq (Insert.insert (Sum.inr b) u).toLeft u.toLeft
                                                                             -/
@[simp] lemma toLeft_insert_inr : (insert (inr b) u).toLeft = u.toLeft := by ext y; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

                                                                                /-
                                                                                  α : Type u_1
                                                                                  β : Type u_2
                                                                                  a : α
                                                                                  u : Finset (Sum α β)
                                                                                  inst✝¹ : DecidableEq α
                                                                                  inst✝ : DecidableEq β
                                                                                  ⊢ Eq (Insert.insert (Sum.inl a) u).toRight u.toRight
                                                                                -/
@[simp] lemma toRight_insert_inl : (insert (inl a) u).toRight = u.toRight := by ext y; simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/

                                                                                         /-
                                                                                           α : Type u_1
                                                                                           β : Type u_2
                                                                                           b : β
                                                                                           u : Finset (Sum α β)
                                                                                           inst✝¹ : DecidableEq α
                                                                                           inst✝ : DecidableEq β
                                                                                           ⊢ Eq (Insert.insert (Sum.inr b) u).toRight (Insert.insert b u.toRight)
                                                                                         -/
@[simp] lemma toRight_insert_inr : (insert (inr b) u).toRight = insert b u.toRight := by ext y; simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  u v : Finset (Sum α β)
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : DecidableEq β
                                                                  ⊢ Eq (Inter.inter u v).toLeft (Inter.inter u.toLeft v.toLeft)
                                                                -/
lemma toLeft_inter : (u ∩ v).toLeft = u.toLeft ∩ v.toLeft := by ext x; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      u v : Finset (Sum α β)
                                                                      inst✝¹ : DecidableEq α
                                                                      inst✝ : DecidableEq β
                                                                      ⊢ Eq (Inter.inter u v).toRight (Inter.inter u.toRight v.toRight)
                                                                    -/
lemma toRight_inter : (u ∩ v).toRight = u.toRight ∩ v.toRight := by ext x; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  u v : Finset (Sum α β)
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : DecidableEq β
                                                                  ⊢ Eq (Union.union u v).toLeft (Union.union u.toLeft v.toLeft)
                                                                -/
lemma toLeft_union : (u ∪ v).toLeft = u.toLeft ∪ v.toLeft := by ext x; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      u v : Finset (Sum α β)
                                                                      inst✝¹ : DecidableEq α
                                                                      inst✝ : DecidableEq β
                                                                      ⊢ Eq (Union.union u v).toRight (Union.union u.toRight v.toRight)
                                                                    -/
lemma toRight_union : (u ∪ v).toRight = u.toRight ∪ v.toRight := by ext x; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  u v : Finset (Sum α β)
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : DecidableEq β
                                                                  ⊢ Eq (SDiff.sdiff u v).toLeft (SDiff.sdiff u.toLeft v.toLeft)
                                                                -/
lemma toLeft_sdiff : (u \ v).toLeft = u.toLeft \ v.toLeft := by ext x; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      u v : Finset (Sum α β)
                                                                      inst✝¹ : DecidableEq α
                                                                      inst✝ : DecidableEq β
                                                                      ⊢ Eq (SDiff.sdiff u v).toRight (SDiff.sdiff u.toRight v.toRight)
                                                                    -/
lemma toRight_sdiff : (u \ v).toRight = u.toRight \ v.toRight := by ext x; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/



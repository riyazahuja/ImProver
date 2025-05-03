/-- Construct an empty or singleton finset from an `Option` -/
def toFinset (o : Option α) : Finset α :=
  o.elim ∅ singleton


@[simp]
theorem toFinset_none : none.toFinset = (∅ : Finset α) :=
  rfl


@[simp]
theorem toFinset_some {a : α} : (some a).toFinset = {a} :=
  rfl


@[simp]
theorem mem_toFinset {a : α} {o : Option α} : a ∈ o.toFinset ↔ a ∈ o := by
  /-
    α : Type u_1
    a : α
    o : Option α
    ⊢ Iff (Membership.mem o.toFinset a) (Membership.mem o a)
  -/
              /-
                🎉 no goals
              -/
  cases o <;> simp [eq_comm]
              /-
                🎉 no goals
              -/


                                                                          /-
                                                                            α : Type u_1
                                                                            o : Option α
                                                                            ⊢ Eq o.toFinset.card (o.elim 0 1)
                                                                          -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
theorem card_toFinset (o : Option α) : o.toFinset.card = o.elim 0 1 := by cases o <;> rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- Given a finset on `α`, lift it to being a finset on `Option α`
using `Option.some` and then insert `Option.none`. -/
def insertNone : Finset α ↪o Finset (Option α) :=
                                                                             /-
                                                                               α : Type u_1
                                                                               β : Type u_2
                                                                               s : Finset α
                                                                               ⊢ Not (Membership.mem (Finset.map Function.Embedding.some s) Option.none)
                                                                             -/
  (OrderEmbedding.ofMapLEIff fun s => cons none (s.map Embedding.some) <| by simp) fun s t => by
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    /-
      α : Type u_1
      β : Type u_2
      s t : Finset α
      ⊢ Iff (LE.le (Finset.cons Option.none (Finset.map Function.Embedding.some s) ⋯ …
    -/
    rw [le_iff_subset, cons_subset_cons, map_subset_map, le_iff_subset]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_insertNone {s : Finset α} : ∀ {o : Option α}, o ∈ insertNone s ↔ ∀ a ∈ o, a ∈ s
                                                                   /-
                                                                     α : Type u_1
                                                                     s : Finset α
                                                                     a : α
                                                                     h : Membership.mem Option.none a
                                                                     ⊢ Membership.mem s a
                                                                   -/
  | none => iff_of_true (Multiset.mem_cons_self _ _) fun a h => by cases h
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                            /-
                                              α : Type u_1
                                              s : Finset α
                                              a : α
                                              ⊢ Iff (Or (Eq (Option.some a) Option.none) (Membership.mem (Finset.map Functio …
                                            -/
  | some a => Multiset.mem_cons.trans <| by simp
                                            /-
                                              🎉 no goals
                                            -/


lemma forall_mem_insertNone {s : Finset α} {p : Option α → Prop} :
                                                            /-
                                                              α : Type u_1
                                                              s : Finset α
                                                              p : Option α → Prop
                                                              ⊢ Iff (∀ (a : Option α), Membership.mem (Finset.insertNone s) a → p a) (And (p …
                                                            -/
    (∀ a ∈ insertNone s, p a) ↔ p none ∧ ∀ a ∈ s, p a := by simp [Option.forall]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                                                         /-
                                                                                           α : Type u_1
                                                                                           s : Finset α
                                                                                           a : α
                                                                                           ⊢ Iff (Membership.mem (Finset.insertNone s) (Option.some a)) (Membership.mem s …
                                                                                         -/
theorem some_mem_insertNone {s : Finset α} {a : α} : some a ∈ insertNone s ↔ a ∈ s := by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


                                                                     /-
                                                                       α : Type u_1
                                                                       s : Finset α
                                                                       ⊢ Membership.mem (Finset.insertNone s) Option.none
                                                                     -/
lemma none_mem_insertNone {s : Finset α} : none ∈ insertNone s := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
lemma insertNone_nonempty {s : Finset α} : insertNone s |>.Nonempty := ⟨none, none_mem_insertNone⟩


@[simp]
                                                                              /-
                                                                                α : Type u_1
                                                                                s : Finset α
                                                                                ⊢ Eq (Finset.insertNone s).card (HAdd.hAdd s.card 1)
                                                                              -/
theorem card_insertNone (s : Finset α) : s.insertNone.card = s.card + 1 := by simp [insertNone]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- Given `s : Finset (Option α)`, `eraseNone s : Finset α` is the set of `x : α` such that
`some x ∈ s`. -/
def eraseNone : Finset (Option α) →o Finset α :=
  (Finset.mapEmbedding (Equiv.optionIsSomeEquiv α).toEmbedding).toOrderHom.comp
    ⟨Finset.subtype _, subtype_mono⟩


@[simp]
theorem mem_eraseNone {s : Finset (Option α)} {x : α} : x ∈ eraseNone s ↔ some x ∈ s := by
  /-
    α : Type u_1
    s : Finset (Option α)
    x : α
    ⊢ Iff (Membership.mem (Finset.eraseNone s) x) (Membership.mem s (Option.some x))
  -/
  simp [eraseNone]
  /-
    🎉 no goals
  -/


lemma forall_mem_eraseNone {s : Finset (Option α)} {p : Option α → Prop} :
                                                                       /-
                                                                         α : Type u_1
                                                                         s : Finset (Option α)
                                                                         p : Option α → Prop
                                                                         ⊢ Iff (∀ (a : α), Membership.mem (Finset.eraseNone s) a → p (Option.some a)) ( …
                                                                       -/
    (∀ a ∈ eraseNone s, p a) ↔ ∀ a : α, (a : Option α) ∈ s → p a := by simp [Option.forall]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem eraseNone_eq_biUnion [DecidableEq α] (s : Finset (Option α)) :
    eraseNone s = s.biUnion Option.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset (Option α)
    ⊢ Eq (Finset.eraseNone s) (s.biUnion Option.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset (Option α)
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone s) a✝) (Membership.mem (s.biUnion Opti …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseNone_map_some (s : Finset α) : eraseNone (s.map Embedding.some) = s := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (Finset.eraseNone (Finset.map Function.Embedding.some s)) s
  -/
  ext
  /-
    case h
    α : Type u_1
    s : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone (Finset.map Function.Embedding.some s) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseNone_image_some [DecidableEq (Option α)] (s : Finset α) :
                                       /-
                                         α : Type u_1
                                         inst✝ : DecidableEq (Option α)
                                         s : Finset α
                                         ⊢ Eq (Finset.eraseNone (Finset.image Option.some s)) s
                                       -/
    eraseNone (s.image some) = s := by simpa only [map_eq_image] using eraseNone_map_some s
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem coe_eraseNone (s : Finset (Option α)) : (eraseNone s : Set α) = some ⁻¹' s :=
  Set.ext fun _ => mem_eraseNone


@[simp]
theorem eraseNone_union [DecidableEq (Option α)] [DecidableEq α] (s t : Finset (Option α)) :
    eraseNone (s ∪ t) = eraseNone s ∪ eraseNone t := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq (Option α)
    inst✝ : DecidableEq α
    s t : Finset (Option α)
    ⊢ Eq (Finset.eraseNone (Union.union s t)) (Union.union (Finset.eraseNone s) (F …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq (Option α)
    inst✝ : DecidableEq α
    s t : Finset (Option α)
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone (Union.union s t)) a✝) (Membership.mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseNone_inter [DecidableEq (Option α)] [DecidableEq α] (s t : Finset (Option α)) :
    eraseNone (s ∩ t) = eraseNone s ∩ eraseNone t := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq (Option α)
    inst✝ : DecidableEq α
    s t : Finset (Option α)
    ⊢ Eq (Finset.eraseNone (Inter.inter s t)) (Inter.inter (Finset.eraseNone s) (F …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq (Option α)
    inst✝ : DecidableEq α
    s t : Finset (Option α)
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone (Inter.inter s t)) a✝) (Membership.mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseNone_empty : eraseNone (∅ : Finset (Option α)) = ∅ := by
  /-
    α : Type u_1
    ⊢ Eq (Finset.eraseNone EmptyCollection.emptyCollection) EmptyCollection.emptyC …
  -/
  ext
  /-
    case h
    α : Type u_1
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone EmptyCollection.emptyCollection) a✝) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eraseNone_none : eraseNone ({none} : Finset (Option α)) = ∅ := by
  /-
    α : Type u_1
    ⊢ Eq (Finset.eraseNone (Singleton.singleton Option.none)) EmptyCollection.empt …
  -/
  ext
  /-
    case h
    α : Type u_1
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone (Singleton.singleton Option.none)) a✝) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem image_some_eraseNone [DecidableEq (Option α)] (s : Finset (Option α)) :
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : DecidableEq (Option α)
                                                    s : Finset (Option α)
                                                    ⊢ Eq (Finset.image Option.some (Finset.eraseNone s)) (s.erase Option.none)
                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (eraseNone s).image some = s.erase none := by ext (_ | x) <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem map_some_eraseNone [DecidableEq (Option α)] (s : Finset (Option α)) :
    (eraseNone s).map Embedding.some = s.erase none := by
  /-
    α : Type u_1
    inst✝ : DecidableEq (Option α)
    s : Finset (Option α)
    ⊢ Eq (Finset.map Function.Embedding.some (Finset.eraseNone s)) (s.erase Option …
  -/
  rw [map_eq_image, Embedding.some_apply, image_some_eraseNone]
  /-
    🎉 no goals
  -/


@[simp]
theorem insertNone_eraseNone [DecidableEq (Option α)] (s : Finset (Option α)) :
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : DecidableEq (Option α)
                                                     s : Finset (Option α)
                                                     ⊢ Eq (Finset.insertNone (Finset.eraseNone s)) (Insert.insert Option.none s)
                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    insertNone (eraseNone s) = insert none s := by ext (_ | x) <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem eraseNone_insertNone (s : Finset α) : eraseNone (insertNone s) = s := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (Finset.eraseNone (Finset.insertNone s)) s
  -/
  ext
  /-
    case h
    α : Type u_1
    s : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (Finset.eraseNone (Finset.insertNone s)) a✝) (Membership …
  -/
  simp
  /-
    🎉 no goals
  -/



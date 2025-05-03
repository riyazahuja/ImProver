/-- `s.WellFoundedOn r` indicates that the relation `r` is well-founded when restricted to `s`. -/
def WellFoundedOn (s : Set α) (r : α → α → Prop) : Prop :=
  WellFounded fun a b : s => r a b


@[simp]
theorem wellFoundedOn_empty (r : α → α → Prop) : WellFoundedOn ∅ r :=
  wellFounded_of_isEmpty _


theorem wellFoundedOn_iff :
    s.WellFoundedOn r ↔ WellFounded fun a b : α => r a b ∧ a ∈ s ∧ b ∈ s := by
  have f : RelEmbedding (fun (a : s) (b : s) => r a b) fun a b : α => r a b ∧ a ∈ s ∧ b ∈ s :=
    ⟨⟨(↑), Subtype.coe_injective⟩, by simp⟩
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
    ⊢ Iff (s.WellFoundedOn r) (WellFounded fun a b => And (r a b) (And (Membership …
  -/
  refine ⟨fun h => ?_, f.wellFounded⟩
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
    h : s.WellFoundedOn r
    ⊢ WellFounded fun a b => And (r a b) (And (Membership.mem s a) (Membership.mem …
  -/
  rw [WellFounded.wellFounded_iff_has_min]
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
    h : s.WellFoundedOn r
    ⊢ ∀ (s_1 : Set α), s_1.Nonempty → Exists fun m => And (Membership.mem s_1 m) ( …
  -/
  intro t ht
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
    h : s.WellFoundedOn r
    t : Set α
    ht : t.Nonempty
    ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
  -/
  by_cases hst : (s ∩ t).Nonempty
    /-
      case pos
      α : Type u_2
      r : α → α → Prop
      s : Set α
      f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
      h : s.WellFoundedOn r
      t : Set α
      ht : t.Nonempty
      hst : (Inter.inter s t).Nonempty
      ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
    -/
  · rw [← Subtype.preimage_coe_nonempty] at hst
    /-
      case pos
      α : Type u_2
      r : α → α → Prop
      s : Set α
      f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
      h : s.WellFoundedOn r
      t : Set α
      ht : t.Nonempty
      hst : (Set.preimage Subtype.val t).Nonempty
      ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
    -/
    rcases h.has_min (Subtype.val ⁻¹' t) hst with ⟨⟨m, ms⟩, mt, hm⟩
    /-
      case pos.intro.mk.intro
      α : Type u_2
      r : α → α → Prop
      s : Set α
      f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
      h : s.WellFoundedOn r
      t : Set α
      ht : t.Nonempty
      hst : (Set.preimage Subtype.val t).Nonempty
      m : α
      ms : Membership.mem s m
      mt : Membership.mem (Set.preimage Subtype.val t) ⟨m, ms⟩
      hm : ∀ (x : ↑s), Membership.mem (Set.preimage Subtype.val t) x → Not (r ↑x ↑⟨m …
      ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
    -/
    exact ⟨m, mt, fun x xt ⟨xm, xs, _⟩ => hm ⟨x, xs⟩ xt xm⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      r : α → α → Prop
      s : Set α
      f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
      h : s.WellFoundedOn r
      t : Set α
      ht : t.Nonempty
      hst : Not (Inter.inter s t).Nonempty
      ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
    -/
  · rcases ht with ⟨m, mt⟩
    /-
      case neg.intro
      α : Type u_2
      r : α → α → Prop
      s : Set α
      f : RelEmbedding (fun a b => r ↑a ↑b) fun a b => And (r a b) (And (Membership. …
      h : s.WellFoundedOn r
      t : Set α
      hst : Not (Inter.inter s t).Nonempty
      m : α
      mt : Membership.mem t m
      ⊢ Exists fun m => And (Membership.mem t m) (∀ (x : α), Membership.mem t x → No …
    -/
    exact ⟨m, mt, fun x _ ⟨_, _, ms⟩ => hst ⟨m, ⟨ms, mt⟩⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem wellFoundedOn_univ : (univ : Set α).WellFoundedOn r ↔ WellFounded r := by
  /-
    α : Type u_2
    r : α → α → Prop
    ⊢ Iff (Set.univ.WellFoundedOn r) (WellFounded r)
  -/
  simp [wellFoundedOn_iff]
  /-
    🎉 no goals
  -/


theorem _root_.WellFounded.wellFoundedOn : WellFounded r → s.WellFoundedOn r :=
  InvImage.wf _


@[simp]
theorem wellFoundedOn_range : (range f).WellFoundedOn r ↔ WellFounded (r on f) := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    f : β → α
    ⊢ Iff ((Set.range f).WellFoundedOn r) (WellFounded (Function.onFun r f))
  -/
  let f' : β → range f := fun c => ⟨f c, c, rfl⟩
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    f : β → α
    f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
    ⊢ Iff ((Set.range f).WellFoundedOn r) (WellFounded (Function.onFun r f))
  -/
  refine ⟨fun h => (InvImage.wf f' h).mono fun c c' => id, fun h => ⟨?_⟩⟩
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    f : β → α
    f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
    h : WellFounded (Function.onFun r f)
    ⊢ ∀ (a : ↑(Set.range f)), Acc (fun a b => r ↑a ↑b) a
  -/
  rintro ⟨_, c, rfl⟩
  /-
    case mk.intro
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    f : β → α
    f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
    h : WellFounded (Function.onFun r f)
    c : β
    ⊢ Acc (fun a b => r ↑a ↑b) ⟨f c, ⋯⟩
  -/
  refine Acc.of_downward_closed f' ?_ _ ?_
    /-
      case mk.intro.refine_1
      α : Type u_2
      β : Type u_3
      r : α → α → Prop
      f : β → α
      f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
      h : WellFounded (Function.onFun r f)
      c : β
      ⊢ ∀ {a : β} {b : ↑(Set.range f)}, r ↑b ↑(f' a) → Exists fun c => Eq (f' c) b
    -/
  · rintro _ ⟨_, c', rfl⟩ -
    /-
      case mk.intro.refine_1.mk.intro
      α : Type u_2
      β : Type u_3
      r : α → α → Prop
      f : β → α
      f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
      h : WellFounded (Function.onFun r f)
      c a✝ c' : β
      ⊢ Exists fun c => Eq (f' c) ⟨f c', ⋯⟩
    -/
    exact ⟨c', rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.refine_2
      α : Type u_2
      β : Type u_3
      r : α → α → Prop
      f : β → α
      f' : β → ↑(Set.range f) := fun c => ⟨f c, ⋯⟩
      h : WellFounded (Function.onFun r f)
      c : β
      ⊢ Acc (InvImage (fun a b => r ↑a ↑b) f') c
    -/
  · exact h.apply _
    /-
      🎉 no goals
    -/


@[simp]
theorem wellFoundedOn_image {s : Set β} : (f '' s).WellFoundedOn r ↔ s.WellFoundedOn (r on f) := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    f : β → α
    s : Set β
    ⊢ Iff ((Set.image f s).WellFoundedOn r) (s.WellFoundedOn (Function.onFun r f))
  -/
  rw [image_eq_range]; exact wellFoundedOn_range
                       /-
                         🎉 no goals
                       -/


protected theorem induction (hs : s.WellFoundedOn r) (hx : x ∈ s) {P : α → Prop}
    (hP : ∀ y ∈ s, (∀ z ∈ s, r z y → P z) → P y) : P x := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    x : α
    hs : s.WellFoundedOn r
    hx : Membership.mem s x
    P : α → Prop
    hP : ∀ (y : α), Membership.mem s y → (∀ (z : α), Membership.mem s z → r z y →  …
    ⊢ P x
  -/
  let Q : s → Prop := fun y => P y
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    x : α
    hs : s.WellFoundedOn r
    hx : Membership.mem s x
    P : α → Prop
    hP : ∀ (y : α), Membership.mem s y → (∀ (z : α), Membership.mem s z → r z y →  …
    Q : ↑s → Prop := fun y => P ↑y
    ⊢ P x
  -/
  change Q ⟨x, hx⟩
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    x : α
    hs : s.WellFoundedOn r
    hx : Membership.mem s x
    P : α → Prop
    hP : ∀ (y : α), Membership.mem s y → (∀ (z : α), Membership.mem s z → r z y →  …
    Q : ↑s → Prop := fun y => P ↑y
    ⊢ Q ⟨x, hx⟩
  -/
  refine WellFounded.induction hs ⟨x, hx⟩ ?_
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    x : α
    hs : s.WellFoundedOn r
    hx : Membership.mem s x
    P : α → Prop
    hP : ∀ (y : α), Membership.mem s y → (∀ (z : α), Membership.mem s z → r z y →  …
    Q : ↑s → Prop := fun y => P ↑y
    ⊢ ∀ (x : ↑s), (∀ (y : ↑s), r ↑y ↑x → Q y) → Q x
  -/
  simpa only [Subtype.forall]
  /-
    🎉 no goals
  -/


protected theorem mono (h : t.WellFoundedOn r') (hle : r ≤ r') (hst : s ⊆ t) :
    s.WellFoundedOn r := by
  /-
    α : Type u_2
    r r' : α → α → Prop
    s t : Set α
    h : t.WellFoundedOn r'
    hle : LE.le r r'
    hst : HasSubset.Subset s t
    ⊢ s.WellFoundedOn r
  -/
  rw [wellFoundedOn_iff] at *
  /-
    α : Type u_2
    r r' : α → α → Prop
    s t : Set α
    h : WellFounded fun a b => And (r' a b) (And (Membership.mem t a) (Membership. …
    hle : LE.le r r'
    hst : HasSubset.Subset s t
    ⊢ WellFounded fun a b => And (r a b) (And (Membership.mem s a) (Membership.mem …
  -/
  exact Subrelation.wf (fun xy => ⟨hle _ _ xy.1, hst xy.2.1, hst xy.2.2⟩) h
  /-
    🎉 no goals
  -/


theorem mono' (h : ∀ (a) (_ : a ∈ s) (b) (_ : b ∈ s), r' a b → r a b) :
    s.WellFoundedOn r → s.WellFoundedOn r' :=
  Subrelation.wf @fun a b => h _ a.2 _ b.2


theorem subset (h : t.WellFoundedOn r) (hst : s ⊆ t) : s.WellFoundedOn r :=
  h.mono le_rfl hst


open List in
/-- `a` is accessible under the relation `r` iff `r` is well-founded on the downward transitive
closure of `a` under `r` (including `a` or not). -/
theorem acc_iff_wellFoundedOn {α} {r : α → α → Prop} {a : α} :
    TFAE [Acc r a,
      WellFoundedOn { b | ReflTransGen r b a } r,
      WellFoundedOn { b | TransGen r b a } r] := by
  tfae_have 1 → 2 := by
    refine fun h => ⟨fun b => InvImage.accessible _ ?_⟩
    rw [← acc_transGen_iff] at h ⊢
    obtain h' | h' := reflTransGen_iff_eq_or_transGen.1 b.2
    · rwa [h'] at h
    · exact h.inv h'
  /-
    α : Type u_6
    r : α → α → Prop
    a : α
    tfae_1_to_2 : Acc r a → (setOf fun b => Relation.ReflTransGen r b a).WellFound …
    ⊢ (List.cons (Acc r a) (List.cons ((setOf fun b => Relation.ReflTransGen r b a …
  -/
  tfae_have 2 → 3 := fun h => h.subset fun _ => TransGen.to_reflTransGen
  tfae_have 3 → 1 := by
    refine fun h => Acc.intro _ (fun b hb => (h.apply ⟨b, .single hb⟩).of_fibration Subtype.val ?_)
    exact fun ⟨c, hc⟩ d h => ⟨⟨d, .head h hc⟩, h, rfl⟩
  /-
    α : Type u_6
    r : α → α → Prop
    a : α
    tfae_1_to_2 : Acc r a → (setOf fun b => Relation.ReflTransGen r b a).WellFound …
    tfae_2_to_3 : (setOf fun b => Relation.ReflTransGen r b a).WellFoundedOn r → ( …
    tfae_3_to_1 : (setOf fun b => Relation.TransGen r b a).WellFoundedOn r → Acc r a
    ⊢ (List.cons (Acc r a) (List.cons ((setOf fun b => Relation.ReflTransGen r b a …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


instance IsStrictOrder.subset : IsStrictOrder α fun a b : α => r a b ∧ a ∈ s ∧ b ∈ s where
  toIsIrrefl := ⟨fun a con => irrefl_of r a con.1⟩
  toIsTrans := ⟨fun _ _ _ ab bc => ⟨trans_of r ab.1 bc.1, ab.2.1, bc.2.2⟩⟩


theorem wellFoundedOn_iff_no_descending_seq :
    s.WellFoundedOn r ↔ ∀ f : ((· > ·) : ℕ → ℕ → Prop) ↪r r, ¬∀ n, f n ∈ s := by
  simp only [wellFoundedOn_iff, RelEmbedding.wellFounded_iff_no_descending_seq, ← not_exists, ←
    not_nonempty_iff, not_iff_not]
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s : Set α
    ⊢ Iff (Nonempty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun a b => And (r a b …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      ⊢ Nonempty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun a b => And (r a b) (An …
    -/
  · rintro ⟨⟨f, hf⟩⟩
    /-
      case mp.intro.mk
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      f : Function.Embedding Nat α
      hf : ∀ {a b : Nat}, Iff (And (r (f a) (f b)) (And (Membership.mem s (f a)) (Me …
      ⊢ Exists fun x => ∀ (n : Nat), Membership.mem s (x n)
    -/
    have H : ∀ n, f n ∈ s := fun n => (hf.2 n.lt_succ_self).2.2
    /-
      case mp.intro.mk
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      f : Function.Embedding Nat α
      hf : ∀ {a b : Nat}, Iff (And (r (f a) (f b)) (And (Membership.mem s (f a)) (Me …
      H : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ Exists fun x => ∀ (n : Nat), Membership.mem s (x n)
    -/
    refine ⟨⟨f, ?_⟩, H⟩
    /-
      case mp.intro.mk
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      f : Function.Embedding Nat α
      hf : ∀ {a b : Nat}, Iff (And (r (f a) (f b)) (And (Membership.mem s (f a)) (Me …
      H : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ ∀ {a b : Nat}, Iff (r (f a) (f b)) (GT.gt a b)
    -/
    simpa only [H, and_true] using @hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      ⊢ (Exists fun x => ∀ (n : Nat), Membership.mem s (x n)) → Nonempty (RelEmbeddi …
    -/
  · rintro ⟨⟨f, hf⟩, hfs : ∀ n, f n ∈ s⟩
    /-
      case mpr.intro.mk
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      f : Function.Embedding Nat α
      hf : ∀ {a b : Nat}, Iff (r (f a) (f b)) (GT.gt a b)
      hfs : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ Nonempty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun a b => And (r a b) (An …
    -/
    refine ⟨⟨f, ?_⟩⟩
    /-
      case mpr.intro.mk
      α : Type u_2
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      s : Set α
      f : Function.Embedding Nat α
      hf : ∀ {a b : Nat}, Iff (r (f a) (f b)) (GT.gt a b)
      hfs : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ ∀ {a b : Nat}, Iff (And (r (f a) (f b)) (And (Membership.mem s (f a)) (Membe …
    -/
    simpa only [hfs, and_true] using @hf
    /-
      🎉 no goals
    -/


theorem WellFoundedOn.union (hs : s.WellFoundedOn r) (ht : t.WellFoundedOn r) :
    (s ∪ t).WellFoundedOn r := by
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s t : Set α
    hs : s.WellFoundedOn r
    ht : t.WellFoundedOn r
    ⊢ (Union.union s t).WellFoundedOn r
  -/
  rw [wellFoundedOn_iff_no_descending_seq] at *
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s t : Set α
    hs : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    ht : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    ⊢ ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Members …
  -/
  rintro f hf
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s t : Set α
    hs : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    ht : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
    ⊢ False
  -/
  rcases Nat.exists_subseq_of_forall_mem_union f hf with ⟨g, hg | hg⟩
  /-
    case intro.inl
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s t : Set α
    hs : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    ht : ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r), Not (∀ (n : Nat), Memb …
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
    g : OrderEmbedding Nat Nat
    hg : ∀ (n : Nat), Membership.mem s (f (g n))
    ⊢ False
  -/
  exacts [hs (g.dual.ltEmbedding.trans f) hg, ht (g.dual.ltEmbedding.trans f) hg]
  /-
    🎉 no goals
  -/


@[simp]
theorem wellFoundedOn_union : (s ∪ t).WellFoundedOn r ↔ s.WellFoundedOn r ∧ t.WellFoundedOn r :=
  ⟨fun h => ⟨h.subset subset_union_left, h.subset subset_union_right⟩, fun h =>
    h.1.union h.2⟩


/-- `s.IsWF` indicates that `<` is well-founded when restricted to `s`. -/
def IsWF (s : Set α) : Prop :=
  WellFoundedOn s (· < ·)


@[simp]
theorem isWF_empty : IsWF (∅ : Set α) :=
  wellFounded_of_isEmpty _


theorem isWF_univ_iff : IsWF (univ : Set α) ↔ WellFounded ((· < ·) : α → α → Prop) := by
  /-
    α : Type u_2
    inst✝ : LT α
    ⊢ Iff Set.univ.IsWF (WellFounded fun x1 x2 => LT.lt x1 x2)
  -/
  simp [IsWF, wellFoundedOn_iff]
  /-
    🎉 no goals
  -/


theorem IsWF.mono (h : IsWF t) (st : s ⊆ t) : IsWF s := h.subset st


protected nonrec theorem IsWF.union (hs : IsWF s) (ht : IsWF t) : IsWF (s ∪ t) := hs.union ht


@[simp] theorem isWF_union : IsWF (s ∪ t) ↔ IsWF s ∧ IsWF t := wellFoundedOn_union


theorem isWF_iff_no_descending_seq :
    IsWF s ↔ ∀ f : ℕ → α, StrictAnti f → ¬∀ n, f (OrderDual.toDual n) ∈ s :=
  wellFoundedOn_iff_no_descending_seq.trans
    ⟨fun H f hf => H ⟨⟨f, hf.injective⟩, hf.lt_iff_lt⟩, fun H f => H f fun _ _ => f.map_rel_iff.2⟩


/-- A subset is partially well-ordered by a relation `r` when any infinite sequence contains
  two elements where the first is related to the second by `r`. -/
def PartiallyWellOrderedOn (s : Set α) (r : α → α → Prop) : Prop :=
  ∀ f : ℕ → α, (∀ n, f n ∈ s) → ∃ m n : ℕ, m < n ∧ r (f m) (f n)


theorem PartiallyWellOrderedOn.mono (ht : t.PartiallyWellOrderedOn r) (h : s ⊆ t) :
    s.PartiallyWellOrderedOn r := fun f hf => ht f fun n => h <| hf n


@[simp]
theorem partiallyWellOrderedOn_empty (r : α → α → Prop) : PartiallyWellOrderedOn ∅ r := fun _ h =>
  (h 0).elim


theorem PartiallyWellOrderedOn.union (hs : s.PartiallyWellOrderedOn r)
    (ht : t.PartiallyWellOrderedOn r) : (s ∪ t).PartiallyWellOrderedOn r := by
  /-
    α : Type u_2
    r : α → α → Prop
    s t : Set α
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r
    ⊢ (Union.union s t).PartiallyWellOrderedOn r
  -/
  rintro f hf
  /-
    α : Type u_2
    r : α → α → Prop
    s t : Set α
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
  -/
  rcases Nat.exists_subseq_of_forall_mem_union f hf with ⟨g, hgs | hgt⟩
    /-
      case intro.inl
      α : Type u_2
      r : α → α → Prop
      s t : Set α
      hs : s.PartiallyWellOrderedOn r
      ht : t.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
      g : OrderEmbedding Nat Nat
      hgs : ∀ (n : Nat), Membership.mem s (f (g n))
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
  · rcases hs _ hgs with ⟨m, n, hlt, hr⟩
    /-
      case intro.inl.intro.intro.intro
      α : Type u_2
      r : α → α → Prop
      s t : Set α
      hs : s.PartiallyWellOrderedOn r
      ht : t.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
      g : OrderEmbedding Nat Nat
      hgs : ∀ (n : Nat), Membership.mem s (f (g n))
      m n : Nat
      hlt : LT.lt m n
      hr : r (f (g m)) (f (g n))
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
    exact ⟨g m, g n, g.strictMono hlt, hr⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_2
      r : α → α → Prop
      s t : Set α
      hs : s.PartiallyWellOrderedOn r
      ht : t.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
      g : OrderEmbedding Nat Nat
      hgt : ∀ (n : Nat), Membership.mem t (f (g n))
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
  · rcases ht _ hgt with ⟨m, n, hlt, hr⟩
    /-
      case intro.inr.intro.intro.intro
      α : Type u_2
      r : α → α → Prop
      s t : Set α
      hs : s.PartiallyWellOrderedOn r
      ht : t.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem (Union.union s t) (f n)
      g : OrderEmbedding Nat Nat
      hgt : ∀ (n : Nat), Membership.mem t (f (g n))
      m n : Nat
      hlt : LT.lt m n
      hr : r (f (g m)) (f (g n))
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
    exact ⟨g m, g n, g.strictMono hlt, hr⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem partiallyWellOrderedOn_union :
    (s ∪ t).PartiallyWellOrderedOn r ↔ s.PartiallyWellOrderedOn r ∧ t.PartiallyWellOrderedOn r :=
  ⟨fun h => ⟨h.mono subset_union_left, h.mono subset_union_right⟩, fun h =>
    h.1.union h.2⟩


theorem PartiallyWellOrderedOn.image_of_monotone_on (hs : s.PartiallyWellOrderedOn r)
    (hf : ∀ a₁ ∈ s, ∀ a₂ ∈ s, r a₁ a₂ → r' (f a₁) (f a₂)) : (f '' s).PartiallyWellOrderedOn r' := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    f : α → β
    s : Set α
    hs : s.PartiallyWellOrderedOn r
    hf : ∀ (a₁ : α), Membership.mem s a₁ → ∀ (a₂ : α), Membership.mem s a₂ → r a₁  …
    ⊢ (Set.image f s).PartiallyWellOrderedOn r'
  -/
  intro g' hg'
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    f : α → β
    s : Set α
    hs : s.PartiallyWellOrderedOn r
    hf : ∀ (a₁ : α), Membership.mem s a₁ → ∀ (a₂ : α), Membership.mem s a₂ → r a₁  …
    g' : Nat → β
    hg' : ∀ (n : Nat), Membership.mem (Set.image f s) (g' n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r' (g' m) (g' n))
  -/
  choose g hgs heq using hg'
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    f : α → β
    s : Set α
    hs : s.PartiallyWellOrderedOn r
    hf : ∀ (a₁ : α), Membership.mem s a₁ → ∀ (a₂ : α), Membership.mem s a₂ → r a₁  …
    g' : Nat → β
    g : Nat → α
    hgs : ∀ (n : Nat), Membership.mem s (g n)
    heq : ∀ (n : Nat), Eq (f (g n)) (g' n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r' (g' m) (g' n))
  -/
  obtain rfl : f ∘ g = g' := funext heq
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    f : α → β
    s : Set α
    hs : s.PartiallyWellOrderedOn r
    hf : ∀ (a₁ : α), Membership.mem s a₁ → ∀ (a₂ : α), Membership.mem s a₂ → r a₁  …
    g : Nat → α
    hgs : ∀ (n : Nat), Membership.mem s (g n)
    heq : ∀ (n : Nat), Eq (f (g n)) (Function.comp f g n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r' (Function.comp f g m) (F …
  -/
  obtain ⟨m, n, hlt, hmn⟩ := hs g hgs
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    f : α → β
    s : Set α
    hs : s.PartiallyWellOrderedOn r
    hf : ∀ (a₁ : α), Membership.mem s a₁ → ∀ (a₂ : α), Membership.mem s a₂ → r a₁  …
    g : Nat → α
    hgs : ∀ (n : Nat), Membership.mem s (g n)
    heq : ∀ (n : Nat), Eq (f (g n)) (Function.comp f g n)
    m n : Nat
    hlt : LT.lt m n
    hmn : r (g m) (g n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r' (Function.comp f g m) (F …
  -/
  exact ⟨m, n, hlt, hf _ (hgs m) _ (hgs n) hmn⟩
  /-
    🎉 no goals
  -/


theorem _root_.IsAntichain.finite_of_partiallyWellOrderedOn (ha : IsAntichain r s)
    (hp : s.PartiallyWellOrderedOn r) : s.Finite := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    ha : IsAntichain r s
    hp : s.PartiallyWellOrderedOn r
    ⊢ s.Finite
  -/
  refine not_infinite.1 fun hi => ?_
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    ha : IsAntichain r s
    hp : s.PartiallyWellOrderedOn r
    hi : s.Infinite
    ⊢ False
  -/
  obtain ⟨m, n, hmn, h⟩ := hp (fun n => hi.natEmbedding _ n) fun n => (hi.natEmbedding _ n).2
  exact hmn.ne ((hi.natEmbedding _).injective <| Subtype.val_injective <|
    ha.eq (hi.natEmbedding _ m).2 (hi.natEmbedding _ n).2 h)


protected theorem Finite.partiallyWellOrderedOn (hs : s.Finite) : s.PartiallyWellOrderedOn r := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsRefl α r
    hs : s.Finite
    ⊢ s.PartiallyWellOrderedOn r
  -/
  intro f hf
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsRefl α r
    hs : s.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
  -/
  obtain ⟨m, n, hmn, h⟩ := hs.exists_lt_map_eq_of_forall_mem hf
  /-
    case intro.intro.intro
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsRefl α r
    hs : s.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    m n : Nat
    hmn : LT.lt m n
    h : Eq (f m) (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
  -/
  exact ⟨m, n, hmn, h.subst <| refl (f m)⟩
  /-
    🎉 no goals
  -/


theorem _root_.IsAntichain.partiallyWellOrderedOn_iff (hs : IsAntichain r s) :
    s.PartiallyWellOrderedOn r ↔ s.Finite :=
  ⟨hs.finite_of_partiallyWellOrderedOn, Finite.partiallyWellOrderedOn⟩


@[simp]
theorem partiallyWellOrderedOn_singleton (a : α) : PartiallyWellOrderedOn {a} r :=
  (finite_singleton a).partiallyWellOrderedOn


@[nontriviality]
theorem Subsingleton.partiallyWellOrderedOn (hs : s.Subsingleton) : PartiallyWellOrderedOn s r :=
  hs.finite.partiallyWellOrderedOn


@[simp]
theorem partiallyWellOrderedOn_insert :
    PartiallyWellOrderedOn (insert a s) r ↔ PartiallyWellOrderedOn s r := by
  simp only [← singleton_union, partiallyWellOrderedOn_union,
    partiallyWellOrderedOn_singleton, true_and]


protected theorem PartiallyWellOrderedOn.insert (h : PartiallyWellOrderedOn s r) (a : α) :
    PartiallyWellOrderedOn (insert a s) r :=
  partiallyWellOrderedOn_insert.2 h


theorem partiallyWellOrderedOn_iff_finite_antichains [IsSymm α r] :
    s.PartiallyWellOrderedOn r ↔ ∀ t, t ⊆ s → IsAntichain r t → t.Finite := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    ⊢ Iff (s.PartiallyWellOrderedOn r) (∀ (t : Set α), HasSubset.Subset t s → IsAn …
  -/
  refine ⟨fun h t ht hrt => hrt.finite_of_partiallyWellOrderedOn (h.mono ht), ?_⟩
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    ⊢ (∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite) → s.Parti …
  -/
  rintro hs f hf
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
  -/
  by_contra! H
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
    ⊢ False
  -/
  refine infinite_range_of_injective (fun m n hmn => ?_) (hs _ (range_subset_iff.2 hf) ?_)
    /-
      case refine_1
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsSymm α r
      hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
      m n : Nat
      hmn : Eq (f m) (f n)
      ⊢ Eq m n
    -/
  · obtain h | h | h := lt_trichotomy m n
      /-
        case refine_1.inl
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt m n
        ⊢ Eq m n
      -/
    · refine (H _ _ h ?_).elim
      /-
        case refine_1.inl
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt m n
        ⊢ r (f m) (f n)
      -/
      rw [hmn]
      /-
        case refine_1.inl
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt m n
        ⊢ r (f n) (f n)
      -/
      exact refl _
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inl
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : Eq m n
        ⊢ Eq m n
      -/
    · exact h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inr
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt n m
        ⊢ Eq m n
      -/
    · refine (H _ _ h ?_).elim
      /-
        case refine_1.inr.inr
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt n m
        ⊢ r (f n) (f m)
      -/
      rw [hmn]
      /-
        case refine_1.inr.inr
        α : Type u_2
        r : α → α → Prop
        s : Set α
        inst✝¹ : IsRefl α r
        inst✝ : IsSymm α r
        hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
        f : Nat → α
        hf : ∀ (n : Nat), Membership.mem s (f n)
        H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
        m n : Nat
        hmn : Eq (f m) (f n)
        h : LT.lt n m
        ⊢ r (f n) (f n)
      -/
      exact refl _
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
    ⊢ IsAntichain r (Set.range f)
  -/
  rintro _ ⟨m, hm, rfl⟩ _ ⟨n, hn, rfl⟩ hmn
  /-
    case refine_2.intro.refl.intro.refl
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsSymm α r
    hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
    m n : Nat
    hmn : Ne (f m) (f n)
    ⊢ HasCompl.compl r (f m) (f n)
  -/
  obtain h | h := (ne_of_apply_ne _ hmn).lt_or_lt
    /-
      case refine_2.intro.refl.intro.refl.inl
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsSymm α r
      hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
      m n : Nat
      hmn : Ne (f m) (f n)
      h : LT.lt m n
      ⊢ HasCompl.compl r (f m) (f n)
    -/
  · exact H _ _ h
    /-
      🎉 no goals
    -/
    /-
      case refine_2.intro.refl.intro.refl.inr
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsSymm α r
      hs : ∀ (t : Set α), HasSubset.Subset t s → IsAntichain r t → t.Finite
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      H : ∀ (m n : Nat), LT.lt m n → Not (r (f m) (f n))
      m n : Nat
      hmn : Ne (f m) (f n)
      h : LT.lt n m
      ⊢ HasCompl.compl r (f m) (f n)
    -/
  · exact mt symm (H _ _ h)
    /-
      🎉 no goals
    -/


theorem PartiallyWellOrderedOn.exists_monotone_subseq (h : s.PartiallyWellOrderedOn r) (f : ℕ → α)
    (hf : ∀ n, f n ∈ s) : ∃ g : ℕ ↪o ℕ, ∀ m n : ℕ, m ≤ n → r (f (g m)) (f (g n)) := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    h : s.PartiallyWellOrderedOn r
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ Exists fun g => ∀ (m n : Nat), LE.le m n → r (f (g m)) (f (g n))
  -/
  obtain ⟨g, h1 | h2⟩ := exists_increasing_or_nonincreasing_subseq r f
    /-
      case intro.inl
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h1 : ∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))
      ⊢ Exists fun g => ∀ (m n : Nat), LE.le m n → r (f (g m)) (f (g n))
    -/
  · refine ⟨g, fun m n hle => ?_⟩
    /-
      case intro.inl
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h1 : ∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))
      m n : Nat
      hle : LE.le m n
      ⊢ r (f (g m)) (f (g n))
    -/
    obtain hlt | rfl := hle.lt_or_eq
    /-
      case intro.inl.inl
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h1 : ∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))
      m n : Nat
      hle : LE.le m n
      hlt : LT.lt m n
      ⊢ r (f (g m)) (f (g n))
    -/
    exacts [h1 m n hlt, refl_of r _]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h2 : ∀ (m n : Nat), LT.lt m n → Not (r (f (g m)) (f (g n)))
      ⊢ Exists fun g => ∀ (m n : Nat), LE.le m n → r (f (g m)) (f (g n))
    -/
  · exfalso
    /-
      case intro.inr
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h2 : ∀ (m n : Nat), LT.lt m n → Not (r (f (g m)) (f (g n)))
      ⊢ False
    -/
    obtain ⟨m, n, hlt, hle⟩ := h (f ∘ g) fun n => hf _
    /-
      case intro.inr.intro.intro.intro
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      h2 : ∀ (m n : Nat), LT.lt m n → Not (r (f (g m)) (f (g n)))
      m n : Nat
      hlt : LT.lt m n
      hle : r (Function.comp f (⇑g) m) (Function.comp f (⇑g) n)
      ⊢ False
    -/
    exact h2 m n hlt hle
    /-
      🎉 no goals
    -/


theorem partiallyWellOrderedOn_iff_exists_monotone_subseq :
    s.PartiallyWellOrderedOn r ↔
      ∀ f : ℕ → α, (∀ n, f n ∈ s) → ∃ g : ℕ ↪o ℕ, ∀ m n : ℕ, m ≤ n → r (f (g m)) (f (g n)) := by
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    ⊢ Iff (s.PartiallyWellOrderedOn r) (∀ (f : Nat → α), (∀ (n : Nat), Membership. …
  -/
  constructor <;> intro h f hf
    /-
      case mp
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : s.PartiallyWellOrderedOn r
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ Exists fun g => ∀ (m n : Nat), LE.le m n → r (f (g m)) (f (g n))
    -/
  · exact h.exists_monotone_subseq f hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem s (f n)) → Exists fun g => ∀ …
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
  · obtain ⟨g, gmon⟩ := h f hf
    /-
      case mpr.intro
      α : Type u_2
      r : α → α → Prop
      s : Set α
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      h : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem s (f n)) → Exists fun g => ∀ …
      f : Nat → α
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      gmon : ∀ (m n : Nat), LE.le m n → r (f (g m)) (f (g n))
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) (r (f m) (f n))
    -/
    exact ⟨g 0, g 1, g.lt_iff_lt.2 Nat.zero_lt_one, gmon _ _ (Nat.zero_le 1)⟩
    /-
      🎉 no goals
    -/


protected theorem PartiallyWellOrderedOn.prod {t : Set β} (hs : PartiallyWellOrderedOn s r)
    (ht : PartiallyWellOrderedOn t r') :
    PartiallyWellOrderedOn (s ×ˢ t) fun x y : α × β => r x.1 y.1 ∧ r' x.2 y.2 := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    t : Set β
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r'
    ⊢ (SProd.sprod s t).PartiallyWellOrderedOn fun x y => And (r x.1 y.1) (r' x.2  …
  -/
  intro f hf
  /-
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    t : Set β
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r'
    f : Nat → Prod α β
    hf : ∀ (n : Nat), Membership.mem (SProd.sprod s t) (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x y => And (r x.1 y.1) …
  -/
  obtain ⟨g₁, h₁⟩ := hs.exists_monotone_subseq (Prod.fst ∘ f) fun n => (hf n).1
  /-
    case intro
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    t : Set β
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r'
    f : Nat → Prod α β
    hf : ∀ (n : Nat), Membership.mem (SProd.sprod s t) (f n)
    g₁ : OrderEmbedding Nat Nat
    h₁ : ∀ (m n : Nat), LE.le m n → r (Function.comp Prod.fst f (g₁ m)) (Function. …
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x y => And (r x.1 y.1) …
  -/
  obtain ⟨m, n, hlt, hle⟩ := ht (Prod.snd ∘ f ∘ g₁) fun n => (hf _).2
  /-
    case intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    t : Set β
    hs : s.PartiallyWellOrderedOn r
    ht : t.PartiallyWellOrderedOn r'
    f : Nat → Prod α β
    hf : ∀ (n : Nat), Membership.mem (SProd.sprod s t) (f n)
    g₁ : OrderEmbedding Nat Nat
    h₁ : ∀ (m n : Nat), LE.le m n → r (Function.comp Prod.fst f (g₁ m)) (Function. …
    m n : Nat
    hlt : LT.lt m n
    hle : r' (Function.comp Prod.snd (Function.comp f ⇑g₁) m) (Function.comp Prod. …
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x y => And (r x.1 y.1) …
  -/
  exact ⟨g₁ m, g₁ n, g₁.strictMono hlt, h₁ _ _ hlt.le, hle⟩
  /-
    🎉 no goals
  -/


theorem PartiallyWellOrderedOn.wellFoundedOn [IsPreorder α r] (h : s.PartiallyWellOrderedOn r) :
    s.WellFoundedOn fun a b => r a b ∧ ¬r b a := by
  letI : Preorder α :=
    { le := r
      le_refl := refl_of r
      le_trans := fun _ _ _ => trans_of r }
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    h : s.PartiallyWellOrderedOn r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    ⊢ s.WellFoundedOn fun a b => And (r a b) (Not (r b a))
  -/
  change s.WellFoundedOn (· < ·)
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    h : s.PartiallyWellOrderedOn r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    ⊢ s.WellFoundedOn fun x1 x2 => LT.lt x1 x2
  -/
  replace h : s.PartiallyWellOrderedOn (· ≤ ·) := h -- Porting note: was `change _ at h`
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    h : s.PartiallyWellOrderedOn fun x1 x2 => LE.le x1 x2
    ⊢ s.WellFoundedOn fun x1 x2 => LT.lt x1 x2
  -/
  rw [wellFoundedOn_iff_no_descending_seq]
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    h : s.PartiallyWellOrderedOn fun x1 x2 => LE.le x1 x2
    ⊢ ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT.lt x1 x2), No …
  -/
  intro f hf
  /-
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    h : s.PartiallyWellOrderedOn fun x1 x2 => LE.le x1 x2
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT.lt x1 x2
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ False
  -/
  obtain ⟨m, n, hlt, hle⟩ := h f hf
  /-
    case intro.intro.intro
    α : Type u_2
    r : α → α → Prop
    s : Set α
    inst✝ : IsPreorder α r
    this : Preorder α := Preorder.mk ⋯ ⋯ ⋯
    h : s.PartiallyWellOrderedOn fun x1 x2 => LE.le x1 x2
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT.lt x1 x2
    hf : ∀ (n : Nat), Membership.mem s (f n)
    m n : Nat
    hlt : LT.lt m n
    hle : LE.le (f m) (f n)
    ⊢ False
  -/
  exact (f.map_rel_iff.2 hlt).not_le hle
  /-
    🎉 no goals
  -/


/-- A subset of a preorder is partially well-ordered when any infinite sequence contains
  a monotone subsequence of length 2 (or equivalently, an infinite monotone subsequence). -/
def IsPWO (s : Set α) : Prop :=
  PartiallyWellOrderedOn s (· ≤ ·)


nonrec theorem IsPWO.mono (ht : t.IsPWO) : s ⊆ t → s.IsPWO := ht.mono


nonrec theorem IsPWO.exists_monotone_subseq (h : s.IsPWO) (f : ℕ → α) (hf : ∀ n, f n ∈ s) :
    ∃ g : ℕ ↪o ℕ, Monotone (f ∘ g) :=
  h.exists_monotone_subseq f hf


theorem isPWO_iff_exists_monotone_subseq :
    s.IsPWO ↔ ∀ f : ℕ → α, (∀ n, f n ∈ s) → ∃ g : ℕ ↪o ℕ, Monotone (f ∘ g) :=
  partiallyWellOrderedOn_iff_exists_monotone_subseq


protected theorem IsPWO.isWF (h : s.IsPWO) : s.IsWF := by
  /-
    α : Type u_2
    inst✝ : Preorder α
    s : Set α
    h : s.IsPWO
    ⊢ s.IsWF
  -/
  simpa only [← lt_iff_le_not_le] using h.wellFoundedOn
  /-
    🎉 no goals
  -/


nonrec theorem IsPWO.prod {t : Set β} (hs : s.IsPWO) (ht : t.IsPWO) : IsPWO (s ×ˢ t) :=
  hs.prod ht


theorem IsPWO.image_of_monotoneOn (hs : s.IsPWO) {f : α → β} (hf : MonotoneOn f s) :
    IsPWO (f '' s) :=
  hs.image_of_monotone_on hf


theorem IsPWO.image_of_monotone (hs : s.IsPWO) {f : α → β} (hf : Monotone f) : IsPWO (f '' s) :=
  hs.image_of_monotone_on (hf.monotoneOn _)


protected nonrec theorem IsPWO.union (hs : IsPWO s) (ht : IsPWO t) : IsPWO (s ∪ t) :=
  hs.union ht


@[simp]
theorem isPWO_union : IsPWO (s ∪ t) ↔ IsPWO s ∧ IsPWO t :=
  partiallyWellOrderedOn_union


protected theorem Finite.isPWO (hs : s.Finite) : IsPWO s := hs.partiallyWellOrderedOn


@[simp] theorem isPWO_of_finite [Finite α] : s.IsPWO := s.toFinite.isPWO


@[simp] theorem isPWO_singleton (a : α) : IsPWO ({a} : Set α) := (finite_singleton a).isPWO


@[simp] theorem isPWO_empty : IsPWO (∅ : Set α) := finite_empty.isPWO


protected theorem Subsingleton.isPWO (hs : s.Subsingleton) : IsPWO s := hs.finite.isPWO


@[simp]
theorem isPWO_insert {a} : IsPWO (insert a s) ↔ IsPWO s := by
  /-
    α : Type u_2
    inst✝ : Preorder α
    s : Set α
    a : α
    ⊢ Iff (Insert.insert a s).IsPWO s.IsPWO
  -/
  simp only [← singleton_union, isPWO_union, isPWO_singleton, true_and]
  /-
    🎉 no goals
  -/


protected theorem IsPWO.insert (h : IsPWO s) (a : α) : IsPWO (insert a s) :=
  isPWO_insert.2 h


protected theorem Finite.isWF (hs : s.Finite) : IsWF s := hs.isPWO.isWF


@[simp] theorem isWF_singleton {a : α} : IsWF ({a} : Set α) := (finite_singleton a).isWF


protected theorem Subsingleton.isWF (hs : s.Subsingleton) : IsWF s := hs.isPWO.isWF


@[simp]
theorem isWF_insert {a} : IsWF (insert a s) ↔ IsWF s := by
  /-
    α : Type u_2
    inst✝ : Preorder α
    s : Set α
    a : α
    ⊢ Iff (Insert.insert a s).IsWF s.IsWF
  -/
  simp only [← singleton_union, isWF_union, isWF_singleton, true_and]
  /-
    🎉 no goals
  -/


protected theorem IsWF.insert (h : IsWF s) (a : α) : IsWF (insert a s) :=
  isWF_insert.2 h


protected theorem Finite.wellFoundedOn (hs : s.Finite) : s.WellFoundedOn r :=
  letI := partialOrderOfSO r
  hs.isWF


@[simp]
theorem wellFoundedOn_singleton : WellFoundedOn ({a} : Set α) r :=
  (finite_singleton a).wellFoundedOn


protected theorem Subsingleton.wellFoundedOn (hs : s.Subsingleton) : s.WellFoundedOn r :=
  hs.finite.wellFoundedOn


@[simp]
theorem wellFoundedOn_insert : WellFoundedOn (insert a s) r ↔ WellFoundedOn s r := by
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s : Set α
    a : α
    ⊢ Iff ((Insert.insert a s).WellFoundedOn r) (s.WellFoundedOn r)
  -/
  simp only [← singleton_union, wellFoundedOn_union, wellFoundedOn_singleton, true_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem wellFoundedOn_sdiff_singleton : WellFoundedOn (s \ {a}) r ↔ WellFoundedOn s r := by
  simp only [← wellFoundedOn_insert (a := a), insert_diff_singleton, mem_insert_iff, true_or,
    insert_eq_of_mem]


protected theorem WellFoundedOn.insert (h : WellFoundedOn s r) (a : α) :
    WellFoundedOn (insert a s) r :=
  wellFoundedOn_insert.2 h


protected theorem WellFoundedOn.sdiff_singleton (h : WellFoundedOn s r) (a : α) :
    WellFoundedOn (s \ {a}) r :=
  wellFoundedOn_sdiff_singleton.2 h


lemma WellFoundedOn.mapsTo {α β : Type*} {r : α → α → Prop} (f : β → α)
    {s : Set α} {t : Set β} (h : MapsTo f t s) (hw : s.WellFoundedOn r) :
    t.WellFoundedOn (r on f) := by
  /-
    α : Type u_6
    β : Type u_7
    r : α → α → Prop
    f : β → α
    s : Set α
    t : Set β
    h : Set.MapsTo f t s
    hw : s.WellFoundedOn r
    ⊢ t.WellFoundedOn (Function.onFun r f)
  -/
  exact InvImage.wf (fun x : t ↦ ⟨f x, h x.prop⟩) hw
  /-
    🎉 no goals
  -/


protected theorem IsWF.isPWO (hs : s.IsWF) : s.IsPWO := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Set α
    hs : s.IsWF
    ⊢ s.IsPWO
  -/
  intro f hf
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s : Set α
    hs : s.IsWF
    f : Nat → α
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  lift f to ℕ → s using hf
  /-
    case intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Set α
    hs : s.IsWF
    f : Nat → ↑s
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  rcases hs.has_min (range f) (range_nonempty _) with ⟨_, ⟨m, rfl⟩, hm⟩
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Set α
    hs : s.IsWF
    f : Nat → ↑s
    m : Nat
    hm : ∀ (x : ↑s), Membership.mem (Set.range f) x → Not ((fun x1 x2 => LT.lt x1  …
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  simp only [forall_mem_range, not_lt] at hm
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝ : LinearOrder α
    s : Set α
    hs : s.IsWF
    f : Nat → ↑s
    m : Nat
    hm : ∀ (i : Nat), LE.le ↑(f m) ↑(f i)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  exact ⟨m, m + 1, by omega, hm _⟩
  /-
    🎉 no goals
  -/


/-- In a linear order, the predicates `Set.IsWF` and `Set.IsPWO` are equivalent. -/
theorem isWF_iff_isPWO : s.IsWF ↔ s.IsPWO :=
  ⟨IsWF.isPWO, IsPWO.isWF⟩


@[simp]
protected theorem partiallyWellOrderedOn [IsRefl α r] (s : Finset α) :
    (s : Set α).PartiallyWellOrderedOn r :=
  s.finite_toSet.partiallyWellOrderedOn


@[simp]
protected theorem isPWO [Preorder α] (s : Finset α) : Set.IsPWO (↑s : Set α) :=
  s.partiallyWellOrderedOn


@[simp]
protected theorem isWF [Preorder α] (s : Finset α) : Set.IsWF (↑s : Set α) :=
  s.finite_toSet.isWF


@[simp]
protected theorem wellFoundedOn [IsStrictOrder α r] (s : Finset α) :
    Set.WellFoundedOn (↑s : Set α) r :=
  letI := partialOrderOfSO r
  s.isWF


theorem wellFoundedOn_sup [IsStrictOrder α r] (s : Finset ι) {f : ι → Set α} :
    (s.sup f).WellFoundedOn r ↔ ∀ i ∈ s, (f i).WellFoundedOn r :=
                                 /-
                                   ι : Type u_1
                                   α : Type u_2
                                   r : α → α → Prop
                                   inst✝ : IsStrictOrder α r
                                   s : Finset ι
                                   f : ι → Set α
                                   ⊢ Iff ((EmptyCollection.emptyCollection.sup f).WellFoundedOn r) (∀ (i : ι), Me …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  Finset.cons_induction_on s (by simp) fun a s ha hs => by simp [-sup_set_eq_biUnion, hs]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem partiallyWellOrderedOn_sup (s : Finset ι) {f : ι → Set α} :
    (s.sup f).PartiallyWellOrderedOn r ↔ ∀ i ∈ s, (f i).PartiallyWellOrderedOn r :=
                                 /-
                                   ι : Type u_1
                                   α : Type u_2
                                   r : α → α → Prop
                                   s : Finset ι
                                   f : ι → Set α
                                   ⊢ Iff ((EmptyCollection.emptyCollection.sup f).PartiallyWellOrderedOn r) (∀ (i …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  Finset.cons_induction_on s (by simp) fun a s ha hs => by simp [-sup_set_eq_biUnion, hs]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem isWF_sup [Preorder α] (s : Finset ι) {f : ι → Set α} :
    (s.sup f).IsWF ↔ ∀ i ∈ s, (f i).IsWF :=
  s.wellFoundedOn_sup


theorem isPWO_sup [Preorder α] (s : Finset ι) {f : ι → Set α} :
    (s.sup f).IsPWO ↔ ∀ i ∈ s, (f i).IsPWO :=
  s.partiallyWellOrderedOn_sup


@[simp]
theorem wellFoundedOn_bUnion [IsStrictOrder α r] (s : Finset ι) {f : ι → Set α} :
    (⋃ i ∈ s, f i).WellFoundedOn r ↔ ∀ i ∈ s, (f i).WellFoundedOn r := by
  /-
    ι : Type u_1
    α : Type u_2
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    s : Finset ι
    f : ι → Set α
    ⊢ Iff ((Set.iUnion fun i => Set.iUnion fun h => f i).WellFoundedOn r) (∀ (i :  …
  -/
  simpa only [Finset.sup_eq_iSup] using s.wellFoundedOn_sup
  /-
    🎉 no goals
  -/


@[simp]
theorem partiallyWellOrderedOn_bUnion (s : Finset ι) {f : ι → Set α} :
    (⋃ i ∈ s, f i).PartiallyWellOrderedOn r ↔ ∀ i ∈ s, (f i).PartiallyWellOrderedOn r := by
  /-
    ι : Type u_1
    α : Type u_2
    r : α → α → Prop
    s : Finset ι
    f : ι → Set α
    ⊢ Iff ((Set.iUnion fun i => Set.iUnion fun h => f i).PartiallyWellOrderedOn r) …
  -/
  simpa only [Finset.sup_eq_iSup] using s.partiallyWellOrderedOn_sup
  /-
    🎉 no goals
  -/


@[simp]
theorem isWF_bUnion [Preorder α] (s : Finset ι) {f : ι → Set α} :
    (⋃ i ∈ s, f i).IsWF ↔ ∀ i ∈ s, (f i).IsWF :=
  s.wellFoundedOn_bUnion


@[simp]
theorem isPWO_bUnion [Preorder α] (s : Finset ι) {f : ι → Set α} :
    (⋃ i ∈ s, f i).IsPWO ↔ ∀ i ∈ s, (f i).IsPWO :=
  s.partiallyWellOrderedOn_bUnion


/-- `Set.IsWF.min` returns a minimal element of a nonempty well-founded set. -/
noncomputable nonrec def IsWF.min (hs : IsWF s) (hn : s.Nonempty) : α :=
  hs.min univ (nonempty_iff_univ_nonempty.1 hn.to_subtype)


theorem IsWF.min_mem (hs : IsWF s) (hn : s.Nonempty) : hs.min hn ∈ s :=
  (WellFounded.min hs univ (nonempty_iff_univ_nonempty.1 hn.to_subtype)).2


nonrec theorem IsWF.not_lt_min (hs : IsWF s) (hn : s.Nonempty) (ha : a ∈ s) : ¬a < hs.min hn :=
  hs.not_lt_min univ (nonempty_iff_univ_nonempty.1 hn.to_subtype) (mem_univ (⟨a, ha⟩ : s))


theorem IsWF.min_of_subset_not_lt_min {hs : s.IsWF} {hsn : s.Nonempty} {ht : t.IsWF}
    {htn : t.Nonempty} (hst : s ⊆ t) : ¬hs.min hsn < ht.min htn :=
  ht.not_lt_min htn (hst (min_mem hs hsn))


@[simp]
theorem isWF_min_singleton (a) {hs : IsWF ({a} : Set α)} {hn : ({a} : Set α).Nonempty} :
    hs.min hn = a :=
  eq_of_mem_singleton (IsWF.min_mem hs hn)


theorem IsWF.min_eq_of_lt (hs : s.IsWF) (ha : a ∈ s) (hlt : ∀ b ∈ s, b ≠ a → a < b) :
    hs.min (nonempty_of_mem ha) = a := by
  /-
    α : Type u_2
    inst✝ : Preorder α
    s : Set α
    a : α
    hs : s.IsWF
    ha : Membership.mem s a
    hlt : ∀ (b : α), Membership.mem s b → Ne b a → LT.lt a b
    ⊢ Eq (hs.min ⋯) a
  -/
  by_contra h
  exact (hs.not_lt_min (nonempty_of_mem ha) ha) (hlt (hs.min (nonempty_of_mem ha))
    (hs.min_mem (nonempty_of_mem ha)) h)


theorem IsWF.min_eq_of_le (hs : s.IsWF) (ha : a ∈ s) (hle : ∀ b ∈ s, a ≤ b) :
    hs.min (nonempty_of_mem ha) = a :=
  (eq_of_le_of_not_lt (hle (hs.min (nonempty_of_mem ha))
    (hs.min_mem (nonempty_of_mem ha))) (hs.not_lt_min (nonempty_of_mem ha) ha)).symm


theorem IsWF.min_le (hs : s.IsWF) (hn : s.Nonempty) (ha : a ∈ s) : hs.min hn ≤ a :=
  le_of_not_lt (hs.not_lt_min hn ha)


theorem IsWF.le_min_iff (hs : s.IsWF) (hn : s.Nonempty) : a ≤ hs.min hn ↔ ∀ b, b ∈ s → a ≤ b :=
  ⟨fun ha _b hb => le_trans ha (hs.min_le hn hb), fun h => h _ (hs.min_mem _)⟩


theorem IsWF.min_le_min_of_subset {hs : s.IsWF} {hsn : s.Nonempty} {ht : t.IsWF} {htn : t.Nonempty}
    (hst : s ⊆ t) : ht.min htn ≤ hs.min hsn :=
  (IsWF.le_min_iff _ _).2 fun _b hb => ht.min_le htn (hst hb)


theorem IsWF.min_union (hs : s.IsWF) (hsn : s.Nonempty) (ht : t.IsWF) (htn : t.Nonempty) :
    (hs.union ht).min (union_nonempty.2 (Or.intro_left _ hsn)) =
      Min.min (hs.min hsn) (ht.min htn) := by
  refine le_antisymm (le_min (IsWF.min_le_min_of_subset subset_union_left)
    (IsWF.min_le_min_of_subset subset_union_right)) ?_
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    s t : Set α
    hs : s.IsWF
    hsn : s.Nonempty
    ht : t.IsWF
    htn : t.Nonempty
    ⊢ LE.le (Min.min (hs.min hsn) (ht.min htn)) (⋯.min ⋯)
  -/
  rw [min_le_iff]
  exact ((mem_union _ _ _).1 ((hs.union ht).min_mem (union_nonempty.2 (.inl hsn)))).imp
    (hs.min_le _) (ht.min_le _)


theorem BddBelow.wellFoundedOn_lt : BddBelow s → s.WellFoundedOn (· < ·) := by
  /-
    α : Type u_2
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ BddBelow s → s.WellFoundedOn fun x1 x2 => LT.lt x1 x2
  -/
  rw [wellFoundedOn_iff_no_descending_seq]
  /-
    α : Type u_2
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    ⊢ BddBelow s → ∀ (f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT. …
  -/
  rintro ⟨a, ha⟩ f hf
  /-
    case intro
    α : Type u_2
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : LocallyFiniteOrder α
    a : α
    ha : Membership.mem (lowerBounds s) a
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT.lt x1 x2
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ False
  -/
  refine infinite_range_of_injective f.injective ?_
  exact (finite_Icc a <| f 0).subset <| range_subset_iff.2 <| fun n =>
    ⟨ha <| hf _,
      antitone_iff_forall_lt.2 (fun a b hab => (f.map_rel_iff.2 hab).le) <| Nat.zero_le _⟩


theorem BddAbove.wellFoundedOn_gt : BddAbove s → s.WellFoundedOn (· > ·) :=
  fun h => h.dual.wellFoundedOn_lt


/-- In the context of partial well-orderings, a bad sequence is a nonincreasing sequence
  whose range is contained in a particular set `s`. One exists if and only if `s` is not
  partially well-ordered. -/
def IsBadSeq (r : α → α → Prop) (s : Set α) (f : ℕ → α) : Prop :=
  (∀ n, f n ∈ s) ∧ ∀ m n : ℕ, m < n → ¬r (f m) (f n)


theorem iff_forall_not_isBadSeq (r : α → α → Prop) (s : Set α) :
    s.PartiallyWellOrderedOn r ↔ ∀ f, ¬IsBadSeq r s f :=
                            /-
                              α : Type u_2
                              r : α → α → Prop
                              s : Set α
                              f : Nat → α
                              ⊢ Iff ((∀ (n : Nat), Membership.mem s (f n)) → Exists fun m => Exists fun n => …
                            -/
  forall_congr' fun f => by simp [IsBadSeq]
                            /-
                              🎉 no goals
                            -/


/-- This indicates that every bad sequence `g` that agrees with `f` on the first `n`
  terms has `rk (f n) ≤ rk (g n)`. -/
def IsMinBadSeq (r : α → α → Prop) (rk : α → ℕ) (s : Set α) (n : ℕ) (f : ℕ → α) : Prop :=
  ∀ g : ℕ → α, (∀ m : ℕ, m < n → f m = g m) → rk (g n) < rk (f n) → ¬IsBadSeq r s g


/-- Given a bad sequence `f`, this constructs a bad sequence that agrees with `f` on the first `n`
  terms and is minimal at `n`.
-/
noncomputable def minBadSeqOfBadSeq (r : α → α → Prop) (rk : α → ℕ) (s : Set α) (n : ℕ) (f : ℕ → α)
    (hf : IsBadSeq r s f) :
    { g : ℕ → α // (∀ m : ℕ, m < n → f m = g m) ∧ IsBadSeq r s g ∧ IsMinBadSeq r rk s n g } := by
  classical
    have h : ∃ (k : ℕ) (g : ℕ → α), (∀ m, m < n → f m = g m) ∧ IsBadSeq r s g ∧ rk (g n) = k :=
      ⟨_, f, fun _ _ => rfl, hf, rfl⟩
    obtain ⟨h1, h2, h3⟩ := Classical.choose_spec (Nat.find_spec h)
    refine ⟨Classical.choose (Nat.find_spec h), h1, by convert h2, fun g hg1 hg2 con => ?_⟩
    refine Nat.find_min h ?_ ⟨g, fun m mn => (h1 m mn).trans (hg1 m mn), con, rfl⟩
    rwa [← h3]


theorem exists_min_bad_of_exists_bad (r : α → α → Prop) (rk : α → ℕ) (s : Set α) :
    (∃ f, IsBadSeq r s f) → ∃ f, IsBadSeq r s f ∧ ∀ n, IsMinBadSeq r rk s n f := by
  /-
    α : Type u_2
    r : α → α → Prop
    rk : α → Nat
    s : Set α
    ⊢ (Exists fun f => Set.PartiallyWellOrderedOn.IsBadSeq r s f) → Exists fun f = …
  -/
  rintro ⟨f0, hf0 : IsBadSeq r s f0⟩
  let fs : ∀ n : ℕ, { f : ℕ → α // IsBadSeq r s f ∧ IsMinBadSeq r rk s n f } := by
    refine Nat.rec ?_ fun n fn => ?_
    · exact ⟨(minBadSeqOfBadSeq r rk s 0 f0 hf0).1, (minBadSeqOfBadSeq r rk s 0 f0 hf0).2.2⟩
    · exact ⟨(minBadSeqOfBadSeq r rk s (n + 1) fn.1 fn.2.1).1,
        (minBadSeqOfBadSeq r rk s (n + 1) fn.1 fn.2.1).2.2⟩
  have h : ∀ m n, m ≤ n → (fs m).1 m = (fs n).1 m := fun m n mn => by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le mn; clear mn
    induction' k with k ih
    · rfl
    · rw [ih, (minBadSeqOfBadSeq r rk s (m + k + 1) (fs (m + k)).1 (fs (m + k)).2.1).2.1 m
        (Nat.lt_succ_iff.2 (Nat.add_le_add_left k.zero_le m))]
      rfl
  /-
    case intro
    α : Type u_2
    r : α → α → Prop
    rk : α → Nat
    s : Set α
    f0 : Nat → α
    hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
    fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
    h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
    ⊢ Exists fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s f) (∀ (n : Nat) …
  -/
  refine ⟨fun n => (fs n).1 n, ⟨fun n => (fs n).2.1.1 n, fun m n mn => ?_⟩, fun n g hg1 hg2 => ?_⟩
    /-
      case intro.refine_1
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f0 : Nat → α
      hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
      fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
      h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
      m n : Nat
      mn : LT.lt m n
      ⊢ Not (r ((fun n => ↑(fs n) n) m) ((fun n => ↑(fs n) n) n))
    -/
  · dsimp
    /-
      case intro.refine_1
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f0 : Nat → α
      hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
      fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
      h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
      m n : Nat
      mn : LT.lt m n
      ⊢ Not (r (↑(fs m) m) (↑(fs n) n))
    -/
    rw [h m n mn.le]
    /-
      case intro.refine_1
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f0 : Nat → α
      hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
      fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
      h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
      m n : Nat
      mn : LT.lt m n
      ⊢ Not (r (↑(fs n) m) (↑(fs n) n))
    -/
    exact (fs n).2.1.2 m n mn
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f0 : Nat → α
      hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
      fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
      h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
      n : Nat
      g : Nat → α
      hg1 : ∀ (m : Nat), LT.lt m n → Eq ((fun n => ↑(fs n) n) m) (g m)
      hg2 : LT.lt (rk (g n)) (rk ((fun n => ↑(fs n) n) n))
      ⊢ Not (Set.PartiallyWellOrderedOn.IsBadSeq r s g)
    -/
  · refine (fs n).2.2 g (fun m mn => ?_) hg2
    /-
      case intro.refine_2
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f0 : Nat → α
      hf0 : Set.PartiallyWellOrderedOn.IsBadSeq r s f0
      fs : (n : Nat) → Subtype fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s …
      h : ∀ (m n : Nat), LE.le m n → Eq (↑(fs m) m) (↑(fs n) m)
      n : Nat
      g : Nat → α
      hg1 : ∀ (m : Nat), LT.lt m n → Eq ((fun n => ↑(fs n) n) m) (g m)
      hg2 : LT.lt (rk (g n)) (rk ((fun n => ↑(fs n) n) n))
      m : Nat
      mn : LT.lt m n
      ⊢ Eq (↑(fs n) m) (g m)
    -/
    rw [← h m n mn.le, ← hg1 m mn]
    /-
      🎉 no goals
    -/


theorem iff_not_exists_isMinBadSeq (rk : α → ℕ) {s : Set α} :
    s.PartiallyWellOrderedOn r ↔ ¬∃ f, IsBadSeq r s f ∧ ∀ n, IsMinBadSeq r rk s n f := by
  /-
    α : Type u_2
    r : α → α → Prop
    rk : α → Nat
    s : Set α
    ⊢ Iff (s.PartiallyWellOrderedOn r) (Not (Exists fun f => And (Set.PartiallyWel …
  -/
  rw [iff_forall_not_isBadSeq, ← not_exists, not_congr]
  /-
    α : Type u_2
    r : α → α → Prop
    rk : α → Nat
    s : Set α
    ⊢ Iff (Exists fun x => Set.PartiallyWellOrderedOn.IsBadSeq r s x) (Exists fun  …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      ⊢ (Exists fun x => Set.PartiallyWellOrderedOn.IsBadSeq r s x) → Exists fun f = …
    -/
  · apply exists_min_bad_of_exists_bad
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      ⊢ (Exists fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq r s f) (∀ (n : Nat …
    -/
  · rintro ⟨f, hf1, -⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      r : α → α → Prop
      rk : α → Nat
      s : Set α
      f : Nat → α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq r s f
      ⊢ Exists fun x => Set.PartiallyWellOrderedOn.IsBadSeq r s x
    -/
    exact ⟨f, hf1⟩
    /-
      🎉 no goals
    -/


/-- Higman's Lemma, which states that for any reflexive, transitive relation `r` which is
  partially well-ordered on a set `s`, the relation `List.SublistForall₂ r` is partially
  well-ordered on the set of lists of elements of `s`. That relation is defined so that
  `List.SublistForall₂ r l₁ l₂` whenever `l₁` related pointwise by `r` to a sublist of `l₂`. -/
theorem partiallyWellOrderedOn_sublistForall₂ (r : α → α → Prop) [IsRefl α r] [IsTrans α r]
    {s : Set α} (h : s.PartiallyWellOrderedOn r) :
    { l : List α | ∀ x, x ∈ l → x ∈ s }.PartiallyWellOrderedOn (List.SublistForall₂ r) := by
  /-
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    ⊢ (setOf fun l => ∀ (x : α), Membership.mem l x → Membership.mem s x).Partiall …
  -/
  rcases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : IsEmpty α
      ⊢ (setOf fun l => ∀ (x : α), Membership.mem l x → Membership.mem s x).Partiall …
    -/
  · exact subsingleton_of_subsingleton.partiallyWellOrderedOn
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    ⊢ (setOf fun l => ∀ (x : α), Membership.mem l x → Membership.mem s x).Partiall …
  -/
  inhabit α
  /-
    case inr
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    ⊢ (setOf fun l => ∀ (x : α), Membership.mem l x → Membership.mem s x).Partiall …
  -/
  rw [iff_not_exists_isMinBadSeq List.length]
  /-
    case inr
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    ⊢ Not (Exists fun f => And (Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistF …
  -/
  rintro ⟨f, hf1, hf2⟩
  have hnil : ∀ n, f n ≠ List.nil := fun n con =>
    hf1.2 n n.succ n.lt_succ_self (con.symm ▸ List.SublistForall₂.nil)
  have : ∀ n, (f n).headI ∈ s :=
    fun n => hf1.1 n _ (List.head!_mem_self (hnil n))
  /-
    case inr.intro.intro
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    f : Nat → List α
    hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
    hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
    hnil : ∀ (n : Nat), Ne (f n) List.nil
    this : ∀ (n : Nat), Membership.mem s (f n).headI
    ⊢ False
  -/
  obtain ⟨g, hg⟩ := h.exists_monotone_subseq (fun n => (f n).headI) this
  have hf' :=
    hf2 (g 0) (fun n => if n < g 0 then f n else List.tail (f (g (n - g 0))))
      (fun m hm => (if_pos hm).symm) ?_
  /-
    case inr.intro.intro.intro.refine_2
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    f : Nat → List α
    hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
    hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
    hnil : ∀ (n : Nat), Ne (f n) List.nil
    this : ∀ (n : Nat), Membership.mem s (f n).headI
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
    hf' : Not (Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf  …
    ⊢ False
  -/
  swap
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      ⊢ LT.lt ((fun n => ite (LT.lt n (g 0)) (f n) (f (g (HSub.hSub n (g 0)))).tail) …
    -/
  · simp only [if_neg (lt_irrefl (g 0)), Nat.sub_self]
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      ⊢ LT.lt (f (g 0)).tail.length (f (g 0)).length
    -/
    rw [List.length_tail, ← Nat.pred_eq_sub_one]
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      ⊢ LT.lt (f (g 0)).length.pred (f (g 0)).length
    -/
    exact Nat.pred_lt fun con => hnil _ (List.length_eq_zero.1 con)
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro.refine_2
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    f : Nat → List α
    hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
    hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
    hnil : ∀ (n : Nat), Ne (f n) List.nil
    this : ∀ (n : Nat), Membership.mem s (f n).headI
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
    hf' : Not (Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf  …
    ⊢ False
  -/
  rw [IsBadSeq] at hf'
  /-
    case inr.intro.intro.intro.refine_2
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    f : Nat → List α
    hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
    hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
    hnil : ∀ (n : Nat), Ne (f n) List.nil
    this : ∀ (n : Nat), Membership.mem s (f n).headI
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
    hf' : Not (And (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Members …
    ⊢ False
  -/
  push_neg at hf'
  obtain ⟨m, n, mn, hmn⟩ := hf' fun n x hx => by
    split_ifs at hx with hn
    exacts [hf1.1 _ _ hx, hf1.1 _ _ (List.tail_subset _ hx)]
  /-
    case inr.intro.intro.intro.refine_2.intro.intro.intro
    α : Type u_2
    r : α → α → Prop
    inst✝¹ : IsRefl α r
    inst✝ : IsTrans α r
    s : Set α
    h : s.PartiallyWellOrderedOn r
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    f : Nat → List α
    hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
    hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
    hnil : ∀ (n : Nat), Ne (f n) List.nil
    this : ∀ (n : Nat), Membership.mem s (f n).headI
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
    hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
    m n : Nat
    mn : LT.lt m n
    hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
    ⊢ False
  -/
  by_cases hn : n < g 0
    /-
      case pos
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
      m n : Nat
      mn : LT.lt m n
      hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
      hn : LT.lt n (g 0)
      ⊢ False
    -/
  · apply hf1.2 m n mn
    /-
      case pos
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
      m n : Nat
      mn : LT.lt m n
      hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
      hn : LT.lt n (g 0)
      ⊢ List.SublistForall₂ r (f m) (f n)
    -/
    rwa [if_pos hn, if_pos (mn.trans hn)] at hmn
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
      m n : Nat
      mn : LT.lt m n
      hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
      hn : Not (LT.lt n (g 0))
      ⊢ False
    -/
  · obtain ⟨n', rfl⟩ := Nat.exists_eq_add_of_le (not_lt.1 hn)
    /-
      case neg.intro
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
      m n' : Nat
      mn : LT.lt m (HAdd.hAdd (g 0) n')
      hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
      hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
      ⊢ False
    -/
    rw [if_neg hn, add_comm (g 0) n', Nat.add_sub_cancel_right] at hmn
    /-
      case neg.intro
      α : Type u_2
      r : α → α → Prop
      inst✝¹ : IsRefl α r
      inst✝ : IsTrans α r
      s : Set α
      h : s.PartiallyWellOrderedOn r
      h✝ : Nonempty α
      inhabited_h : Inhabited α
      f : Nat → List α
      hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
      hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
      hnil : ∀ (n : Nat), Ne (f n) List.nil
      this : ∀ (n : Nat), Membership.mem s (f n).headI
      g : OrderEmbedding Nat Nat
      hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
      hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
      m n' : Nat
      mn : LT.lt m (HAdd.hAdd (g 0) n')
      hmn : List.SublistForall₂ r (ite (LT.lt m (g 0)) (f m) (f (g (HSub.hSub m (g 0 …
      hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
      ⊢ False
    -/
    split_ifs at hmn with hm
      /-
        case pos
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt m (HAdd.hAdd (g 0) n')
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : LT.lt m (g 0)
        hmn : List.SublistForall₂ r (f m) (f (g n')).tail
        ⊢ False
      -/
    · apply hf1.2 m (g n') (lt_of_lt_of_le hm (g.monotone n'.zero_le))
      /-
        case pos
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt m (HAdd.hAdd (g 0) n')
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : LT.lt m (g 0)
        hmn : List.SublistForall₂ r (f m) (f (g n')).tail
        ⊢ List.SublistForall₂ r (f m) (f (g n'))
      -/
      exact _root_.trans hmn (List.tail_sublistForall₂_self _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt m (HAdd.hAdd (g 0) n')
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : Not (LT.lt m (g 0))
        hmn : List.SublistForall₂ r (f (g (HSub.hSub m (g 0)))).tail (f (g n')).tail
        ⊢ False
      -/
    · rw [← Nat.sub_lt_iff_lt_add (le_of_not_lt hm)] at mn
      /-
        case neg
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt (HSub.hSub m (g 0)) n'
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : Not (LT.lt m (g 0))
        hmn : List.SublistForall₂ r (f (g (HSub.hSub m (g 0)))).tail (f (g n')).tail
        ⊢ False
      -/
      apply hf1.2 _ _ (g.lt_iff_lt.2 mn)
      /-
        case neg
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt (HSub.hSub m (g 0)) n'
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : Not (LT.lt m (g 0))
        hmn : List.SublistForall₂ r (f (g (HSub.hSub m (g 0)))).tail (f (g n')).tail
        ⊢ List.SublistForall₂ r (f (g (HSub.hSub m (g 0)))) (f (g n'))
      -/
      rw [← List.cons_head!_tail (hnil (g (m - g 0))), ← List.cons_head!_tail (hnil (g n'))]
      /-
        case neg
        α : Type u_2
        r : α → α → Prop
        inst✝¹ : IsRefl α r
        inst✝ : IsTrans α r
        s : Set α
        h : s.PartiallyWellOrderedOn r
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : Nat → List α
        hf1 : Set.PartiallyWellOrderedOn.IsBadSeq (List.SublistForall₂ r) (setOf fun l …
        hf2 : ∀ (n : Nat), Set.PartiallyWellOrderedOn.IsMinBadSeq (List.SublistForall₂ …
        hnil : ∀ (n : Nat), Ne (f n) List.nil
        this : ∀ (n : Nat), Membership.mem s (f n).headI
        g : OrderEmbedding Nat Nat
        hg : ∀ (m n : Nat), LE.le m n → r (f (g m)).headI (f (g n)).headI
        hf' : (∀ (n : Nat), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l …
        m n' : Nat
        mn : LT.lt (HSub.hSub m (g 0)) n'
        hn : Not (LT.lt (HAdd.hAdd (g 0) n') (g 0))
        hm : Not (LT.lt m (g 0))
        hmn : List.SublistForall₂ r (f (g (HSub.hSub m (g 0)))).tail (f (g n')).tail
        ⊢ List.SublistForall₂ r (List.cons (f (g (HSub.hSub m (g 0)))).head! (f (g (HS …
      -/
      exact List.SublistForall₂.cons (hg _ _ (le_of_lt mn)) hmn
      /-
        🎉 no goals
      -/


theorem subsetProdLex [PartialOrder α] [Preorder β] {s : Set (α ×ₗ β)}
    (hα : ((fun (x : α ×ₗ β) => (ofLex x).1)'' s).IsPWO)
    (hβ : ∀ a, {y | toLex (a, y) ∈ s}.IsPWO) : s.IsPWO := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hα : (Set.image (fun x => (ofLex x).1) s).IsPWO
    hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
    ⊢ s.IsPWO
  -/
  intro f hf
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hα : (Set.image (fun x => (ofLex x).1) s).IsPWO
    hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
    f : Nat → Lex (Prod α β)
    hf : ∀ (n : Nat), Membership.mem s (f n)
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  rw [isPWO_iff_exists_monotone_subseq] at hα
  obtain ⟨g, hg⟩ : ∃ (g : (ℕ ↪o ℕ)), Monotone fun n => (ofLex f (g n)).1 :=
    hα (fun n => (ofLex f n).1) (fun k => mem_image_of_mem (fun x => (ofLex x).1) (hf k))
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
    hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
    f : Nat → Lex (Prod α β)
    hf : ∀ (n : Nat), Membership.mem s (f n)
    g : OrderEmbedding Nat Nat
    hg : Monotone fun n => (ofLex f (g n)).1
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  have hhg : ∀ n, (ofLex f (g 0)).1 ≤ (ofLex f (g n)).1 := fun n => hg n.zero_le
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
    hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
    f : Nat → Lex (Prod α β)
    hf : ∀ (n : Nat), Membership.mem s (f n)
    g : OrderEmbedding Nat Nat
    hg : Monotone fun n => (ofLex f (g n)).1
    hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
    ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
  -/
  by_cases hc : ∃ n, (ofLex f (g 0)).1 < (ofLex f (g n)).1
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      s : Set (Lex (Prod α β))
      hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
      hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
      f : Nat → Lex (Prod α β)
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      hg : Monotone fun n => (ofLex f (g n)).1
      hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
      hc : Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
    -/
  · obtain ⟨n, hn⟩ := hc
    /-
      case pos.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      s : Set (Lex (Prod α β))
      hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
      hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
      f : Nat → Lex (Prod α β)
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      hg : Monotone fun n => (ofLex f (g n)).1
      hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
      n : Nat
      hn : LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
    -/
    use (g 0), (g n)
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      s : Set (Lex (Prod α β))
      hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
      hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
      f : Nat → Lex (Prod α β)
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      hg : Monotone fun n => (ofLex f (g n)).1
      hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
      n : Nat
      hn : LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
      ⊢ And (LT.lt (g 0) (g n)) ((fun x1 x2 => LE.le x1 x2) (f (g 0)) (f (g n)))
    -/
    constructor
      /-
        case h.left
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        n : Nat
        hn : LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
        ⊢ LT.lt (g 0) (g n)
      -/
    · by_contra hx
      /-
        case h.left
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        n : Nat
        hn : LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
        hx : Not (LT.lt (g 0) (g n))
        ⊢ False
      -/
      simp_all
      /-
        🎉 no goals
      -/
      /-
        case h.right
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        n : Nat
        hn : LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1
        ⊢ (fun x1 x2 => LE.le x1 x2) (f (g 0)) (f (g n))
      -/
    · exact (Prod.Lex.le_iff (f (g 0)) _).mpr <| Or.inl hn
      /-
        🎉 no goals
      -/
  · have hhc : ∀ n, (ofLex f (g 0)).1 = (ofLex f (g n)).1 := by
      intro n
      rw [not_exists] at hc
      exact (hhg n).eq_of_not_lt (hc n)
    obtain ⟨g', hg'⟩ : ∃ g' : ℕ ↪o ℕ, Monotone ((fun n ↦ (ofLex f (g (g' n))).2)) := by
      simp_rw [isPWO_iff_exists_monotone_subseq] at hβ
      apply hβ (ofLex f (g 0)).1 fun n ↦ (ofLex f (g n)).2
      intro n
      rw [hhc n]
      simpa using hf _
    /-
      case neg.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      s : Set (Lex (Prod α β))
      hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
      hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
      f : Nat → Lex (Prod α β)
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      hg : Monotone fun n => (ofLex f (g n)).1
      hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
      hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
      hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
      g' : OrderEmbedding Nat Nat
      hg' : Monotone fun n => (ofLex f (g (g' n))).2
      ⊢ Exists fun m => Exists fun n => And (LT.lt m n) ((fun x1 x2 => LE.le x1 x2)  …
    -/
    use (g (g' 0)), (g (g' 1))
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : PartialOrder α
      inst✝ : Preorder β
      s : Set (Lex (Prod α β))
      hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
      hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
      f : Nat → Lex (Prod α β)
      hf : ∀ (n : Nat), Membership.mem s (f n)
      g : OrderEmbedding Nat Nat
      hg : Monotone fun n => (ofLex f (g n)).1
      hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
      hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
      hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
      g' : OrderEmbedding Nat Nat
      hg' : Monotone fun n => (ofLex f (g (g' n))).2
      ⊢ And (LT.lt (g (g' 0)) (g (g' 1))) ((fun x1 x2 => LE.le x1 x2) (f (g (g' 0))) …
    -/
    suffices (f (g (g' 0))) ≤ (f (g (g' 1))) by simpa
      /-
        case h
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
        hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
        g' : OrderEmbedding Nat Nat
        hg' : Monotone fun n => (ofLex f (g (g' n))).2
        ⊢ LE.le (f (g (g' 0))) (f (g (g' 1)))
      -/
    · refine (Prod.Lex.le_iff (f (g (g' 0))) (f (g (g' 1)))).mpr ?_
      /-
        case h
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
        hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
        g' : OrderEmbedding Nat Nat
        hg' : Monotone fun n => (ofLex f (g (g' n))).2
        ⊢ Or (LT.lt (f (g (g' 0))).1 (f (g (g' 1))).1) (And (Eq (f (g (g' 0))).1 (f (g …
      -/
      right
      /-
        case h.h
        α : Type u_2
        β : Type u_3
        inst✝¹ : PartialOrder α
        inst✝ : Preorder β
        s : Set (Lex (Prod α β))
        hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
        hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
        f : Nat → Lex (Prod α β)
        hf : ∀ (n : Nat), Membership.mem s (f n)
        g : OrderEmbedding Nat Nat
        hg : Monotone fun n => (ofLex f (g n)).1
        hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
        hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
        hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
        g' : OrderEmbedding Nat Nat
        hg' : Monotone fun n => (ofLex f (g (g' n))).2
        ⊢ And (Eq (f (g (g' 0))).1 (f (g (g' 1))).1) (LE.le (f (g (g' 0))).2 (f (g (g' …
      -/
      constructor
        /-
          case h.h.left
          α : Type u_2
          β : Type u_3
          inst✝¹ : PartialOrder α
          inst✝ : Preorder β
          s : Set (Lex (Prod α β))
          hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
          hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
          f : Nat → Lex (Prod α β)
          hf : ∀ (n : Nat), Membership.mem s (f n)
          g : OrderEmbedding Nat Nat
          hg : Monotone fun n => (ofLex f (g n)).1
          hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
          hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
          hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
          g' : OrderEmbedding Nat Nat
          hg' : Monotone fun n => (ofLex f (g (g' n))).2
          ⊢ Eq (f (g (g' 0))).1 (f (g (g' 1))).1
        -/
      · exact (hhc (g' 0)).symm.trans (hhc (g' 1))
        /-
          🎉 no goals
        -/
        /-
          case h.h.right
          α : Type u_2
          β : Type u_3
          inst✝¹ : PartialOrder α
          inst✝ : Preorder β
          s : Set (Lex (Prod α β))
          hα : ∀ (f : Nat → α), (∀ (n : Nat), Membership.mem (Set.image (fun x => (ofLex …
          hβ : ∀ (a : α), (setOf fun y => Membership.mem s (toLex { fst := a, snd := y } …
          f : Nat → Lex (Prod α β)
          hf : ∀ (n : Nat), Membership.mem s (f n)
          g : OrderEmbedding Nat Nat
          hg : Monotone fun n => (ofLex f (g n)).1
          hhg : ∀ (n : Nat), LE.le (ofLex f (g 0)).1 (ofLex f (g n)).1
          hc : Not (Exists fun n => LT.lt (ofLex f (g 0)).1 (ofLex f (g n)).1)
          hhc : ∀ (n : Nat), Eq (ofLex f (g 0)).1 (ofLex f (g n)).1
          g' : OrderEmbedding Nat Nat
          hg' : Monotone fun n => (ofLex f (g (g' n))).2
          ⊢ LE.le (f (g (g' 0))).2 (f (g (g' 1))).2
        -/
      · exact hg' (Nat.zero_le 1)
        /-
          🎉 no goals
        -/


theorem imageProdLex [PartialOrder α] [Preorder β] {s : Set (α ×ₗ β)}
    (hαβ : s.IsPWO) : ((fun (x : α ×ₗ β) => (ofLex x).1)'' s).IsPWO :=
  IsPWO.image_of_monotone hαβ Prod.Lex.monotone_fst


theorem fiberProdLex [PartialOrder α] [Preorder β] {s : Set (α ×ₗ β)}
    (hαβ : s.IsPWO) (a : α) : {y | toLex (a, y) ∈ s}.IsPWO := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    ⊢ (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })).IsPWO
  -/
  let f : α ×ₗ β → β := fun x => (ofLex x).2
  have h : {y | toLex (a, y) ∈ s} = f '' (s ∩ (fun x ↦ (ofLex x).1) ⁻¹' {a}) := by
    ext x
    simp [f]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    f : Lex (Prod α β) → β := fun x => (ofLex x).2
    h : Eq (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })) (Set.i …
    ⊢ (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })).IsPWO
  -/
  rw [h]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    f : Lex (Prod α β) → β := fun x => (ofLex x).2
    h : Eq (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })) (Set.i …
    ⊢ (Set.image f (Inter.inter s (Set.preimage (fun x => (ofLex x).1) (Singleton. …
  -/
  apply IsPWO.image_of_monotoneOn (hαβ.mono inter_subset_left)
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    f : Lex (Prod α β) → β := fun x => (ofLex x).2
    h : Eq (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })) (Set.i …
    ⊢ MonotoneOn f (Inter.inter s (Set.preimage (fun x => (ofLex x).1) (Singleton. …
  -/
  rintro b ⟨-, hb⟩ c ⟨-, hc⟩ hbc
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    f : Lex (Prod α β) → β := fun x => (ofLex x).2
    h : Eq (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })) (Set.i …
    b : Lex (Prod α β)
    hb : Membership.mem (Set.preimage (fun x => (ofLex x).1) (Singleton.singleton  …
    c : Lex (Prod α β)
    hc : Membership.mem (Set.preimage (fun x => (ofLex x).1) (Singleton.singleton  …
    hbc : LE.le b c
    ⊢ LE.le (f b) (f c)
  -/
  simp only [mem_preimage, mem_singleton_iff] at hb hc
  have : (ofLex b).1 < (ofLex c).1 ∨ (ofLex b).1 = (ofLex c).1 ∧ f b ≤ f c :=
    (Prod.Lex.le_iff b c).mp hbc
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    s : Set (Lex (Prod α β))
    hαβ : s.IsPWO
    a : α
    f : Lex (Prod α β) → β := fun x => (ofLex x).2
    h : Eq (setOf fun y => Membership.mem s (toLex { fst := a, snd := y })) (Set.i …
    b c : Lex (Prod α β)
    hbc : LE.le b c
    hb : Eq (ofLex b).1 a
    hc : Eq (ofLex c).1 a
    this : Or (LT.lt (ofLex b).1 (ofLex c).1) (And (Eq (ofLex b).1 (ofLex c).1) (L …
    ⊢ LE.le (f b) (f c)
  -/
  simp_all only [lt_self_iff_false, true_and, false_or]
  /-
    🎉 no goals
  -/


theorem ProdLex_iff [PartialOrder α] [Preorder β] {s : Set (α ×ₗ β)} :
    s.IsPWO ↔
      ((fun (x : α ×ₗ β) ↦ (ofLex x).1) '' s).IsPWO ∧ ∀ a, {y | toLex (a, y) ∈ s}.IsPWO :=
  ⟨fun h ↦ ⟨imageProdLex h, fiberProdLex h⟩, fun h ↦ subsetProdLex h.1 h.2⟩


theorem WellFounded.isWF [LT α] (h : WellFounded ((· < ·) : α → α → Prop)) (s : Set α) : s.IsWF :=
  (Set.isWF_univ_iff.2 h).mono s.subset_univ


/-- A version of **Dickson's lemma** any subset of functions `Π s : σ, α s` is partially well
ordered, when `σ` is a `Fintype` and each `α s` is a linear well order.
This includes the classical case of Dickson's lemma that `ℕ ^ n` is a well partial order.
Some generalizations would be possible based on this proof, to include cases where the target is
partially well ordered, and also to consider the case of `Set.PartiallyWellOrderedOn` instead of
`Set.IsPWO`. -/
theorem Pi.isPWO {α : ι → Type*} [∀ i, LinearOrder (α i)] [∀ i, IsWellOrder (α i) (· < ·)]
    [Finite ι] (s : Set (∀ i, α i)) : s.IsPWO := by
  /-
    ι : Type u_1
    α : ι → Type u_6
    inst✝² : (i : ι) → LinearOrder (α i)
    inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
    inst✝ : Finite ι
    s : Set ((i : ι) → α i)
    ⊢ s.IsPWO
  -/
  cases nonempty_fintype ι
  suffices ∀ (s : Finset ι) (f : ℕ → ∀ s, α s),
    ∃ g : ℕ ↪o ℕ, ∀ ⦃a b : ℕ⦄, a ≤ b → ∀ x, x ∈ s → (f ∘ g) a x ≤ (f ∘ g) b x by
    refine isPWO_iff_exists_monotone_subseq.2 fun f _ => ?_
    simpa only [Finset.mem_univ, true_imp_iff] using this Finset.univ f
  /-
    case intro
    ι : Type u_1
    α : ι → Type u_6
    inst✝² : (i : ι) → LinearOrder (α i)
    inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
    inst✝ : Finite ι
    s : Set ((i : ι) → α i)
    val✝ : Fintype ι
    ⊢ ∀ (s : Finset ι) (f : Nat → (s : ι) → α s), Exists fun g => ∀ ⦃a b : Nat⦄, L …
  -/
  refine Finset.cons_induction ?_ ?_
    /-
      case intro.refine_1
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s : Set ((i : ι) → α i)
      val✝ : Fintype ι
      ⊢ ∀ (f : Nat → (s : ι) → α s), Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x …
    -/
  · intro f
    /-
      case intro.refine_1
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s : Set ((i : ι) → α i)
      val✝ : Fintype ι
      f : Nat → (s : ι) → α s
      ⊢ Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x : ι), Membership.mem EmptyCo …
    -/
    exists RelEmbedding.refl (· ≤ ·)
    /-
      case intro.refine_1
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s : Set ((i : ι) → α i)
      val✝ : Fintype ι
      f : Nat → (s : ι) → α s
      ⊢ ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x : ι), Membership.mem EmptyCollection.emptyCo …
    -/
    simp only [IsEmpty.forall_iff, imp_true_iff, forall_const, Finset.not_mem_empty]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s : Set ((i : ι) → α i)
      val✝ : Fintype ι
      ⊢ ∀ (a : ι) (s : Finset ι) (h : Not (Membership.mem s a)), (∀ (f : Nat → (s :  …
    -/
  · intro x s hx ih f
    obtain ⟨g, hg⟩ :=
      (IsWellFounded.wf.isWF univ).isPWO.exists_monotone_subseq (fun n => f n x) mem_univ
    /-
      case intro.refine_2.intro
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s✝ : Set ((i : ι) → α i)
      val✝ : Fintype ι
      x : ι
      s : Finset ι
      hx : Not (Membership.mem s x)
      ih : ∀ (f : Nat → (s : ι) → α s), Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ …
      f : Nat → (s : ι) → α s
      g : OrderEmbedding Nat Nat
      hg : Monotone (Function.comp (fun n => f n x) ⇑g)
      ⊢ Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x_1 : ι), Membership.mem (Fins …
    -/
    obtain ⟨g', hg'⟩ := ih (f ∘ g)
    /-
      case intro.refine_2.intro.intro
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s✝ : Set ((i : ι) → α i)
      val✝ : Fintype ι
      x : ι
      s : Finset ι
      hx : Not (Membership.mem s x)
      ih : ∀ (f : Nat → (s : ι) → α s), Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ …
      f : Nat → (s : ι) → α s
      g : OrderEmbedding Nat Nat
      hg : Monotone (Function.comp (fun n => f n x) ⇑g)
      g' : OrderEmbedding Nat Nat
      hg' : ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x : ι), Membership.mem s x → LE.le (Functi …
      ⊢ Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x_1 : ι), Membership.mem (Fins …
    -/
    refine ⟨g'.trans g, fun a b hab => (Finset.forall_mem_cons _ _).2 ?_⟩
    /-
      case intro.refine_2.intro.intro
      ι : Type u_1
      α : ι → Type u_6
      inst✝² : (i : ι) → LinearOrder (α i)
      inst✝¹ : ∀ (i : ι), IsWellOrder (α i) fun x1 x2 => LT.lt x1 x2
      inst✝ : Finite ι
      s✝ : Set ((i : ι) → α i)
      val✝ : Fintype ι
      x : ι
      s : Finset ι
      hx : Not (Membership.mem s x)
      ih : ∀ (f : Nat → (s : ι) → α s), Exists fun g => ∀ ⦃a b : Nat⦄, LE.le a b → ∀ …
      f : Nat → (s : ι) → α s
      g : OrderEmbedding Nat Nat
      hg : Monotone (Function.comp (fun n => f n x) ⇑g)
      g' : OrderEmbedding Nat Nat
      hg' : ∀ ⦃a b : Nat⦄, LE.le a b → ∀ (x : ι), Membership.mem s x → LE.le (Functi …
      a b : Nat
      hab : LE.le a b
      ⊢ And (LE.le (Function.comp f (⇑(RelEmbedding.trans g' g)) a x) (Function.comp …
    -/
    exact ⟨hg (OrderHomClass.mono g' hab), hg' hab⟩
    /-
      🎉 no goals
    -/


/-- Stronger version of `WellFounded.prod_lex`. Instead of requiring `rβ on g` to be well-founded,
we only require it to be well-founded on fibers of `f`. -/
theorem WellFounded.prod_lex_of_wellFoundedOn_fiber (hα : WellFounded (rα on f))
    (hβ : ∀ a, (f ⁻¹' {a}).WellFoundedOn (rβ on g)) :
    WellFounded (Prod.Lex rα rβ on fun c => (f c, g c)) := by
  refine ((psigma_lex (wellFoundedOn_range.2 hα) fun a => hβ a).onFun
    (f := fun c => ⟨⟨_, c, rfl⟩, c, rfl⟩)).mono fun c c' h => ?_
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    rα : α → α → Prop
    rβ : β → β → Prop
    f : γ → α
    g : γ → β
    hα : WellFounded (Function.onFun rα f)
    hβ : ∀ (a : α), (Set.preimage f (Singleton.singleton a)).WellFoundedOn (Functi …
    c c' : γ
    h : Function.onFun (Prod.Lex rα rβ) (fun c => { fst := f c, snd := g c }) c c'
    ⊢ Function.onFun (PSigma.Lex (fun a b => rα ↑a ↑b) fun a a_1 b => Function.onF …
  -/
  obtain h' | h' := Prod.lex_iff.1 h
    /-
      case inl
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      rα : α → α → Prop
      rβ : β → β → Prop
      f : γ → α
      g : γ → β
      hα : WellFounded (Function.onFun rα f)
      hβ : ∀ (a : α), (Set.preimage f (Singleton.singleton a)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Prod.Lex rα rβ) (fun c => { fst := f c, snd := g c }) c c'
      h' : rα ((fun c => { fst := f c, snd := g c }) c).1 ((fun c => { fst := f c, s …
      ⊢ Function.onFun (PSigma.Lex (fun a b => rα ↑a ↑b) fun a a_1 b => Function.onF …
    -/
  · exact PSigma.Lex.left _ _ h'
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      rα : α → α → Prop
      rβ : β → β → Prop
      f : γ → α
      g : γ → β
      hα : WellFounded (Function.onFun rα f)
      hβ : ∀ (a : α), (Set.preimage f (Singleton.singleton a)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Prod.Lex rα rβ) (fun c => { fst := f c, snd := g c }) c c'
      h' : And (Eq ((fun c => { fst := f c, snd := g c }) c).1 ((fun c => { fst := f …
      ⊢ Function.onFun (PSigma.Lex (fun a b => rα ↑a ↑b) fun a a_1 b => Function.onF …
    -/
  · dsimp only [InvImage, (· on ·)] at h' ⊢
    /-
      case inr
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      rα : α → α → Prop
      rβ : β → β → Prop
      f : γ → α
      g : γ → β
      hα : WellFounded (Function.onFun rα f)
      hβ : ∀ (a : α), (Set.preimage f (Singleton.singleton a)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Prod.Lex rα rβ) (fun c => { fst := f c, snd := g c }) c c'
      h' : And (Eq (f c) (f c')) (rβ (g c) (g c'))
      ⊢ PSigma.Lex (fun a b => rα ↑a ↑b) (fun a a_1 b => rβ (g ↑a_1) (g ↑b)) ⟨⟨f c,  …
    -/
    convert PSigma.Lex.right (⟨_, c', rfl⟩ : range f) _ using 1; swap
    /-
      case inr.convert_4
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      rα : α → α → Prop
      rβ : β → β → Prop
      f : γ → α
      g : γ → β
      hα : WellFounded (Function.onFun rα f)
      hβ : ∀ (a : α), (Set.preimage f (Singleton.singleton a)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Prod.Lex rα rβ) (fun c => { fst := f c, snd := g c }) c c'
      h' : And (Eq (f c) (f c')) (rβ (g c) (g c'))
      ⊢ ↑(Set.preimage f (Singleton.singleton ↑⟨f c', ⋯⟩))
    -/
    exacts [⟨c, h'.1⟩, PSigma.subtype_ext (Subtype.ext h'.1) rfl, h'.2]
    /-
      🎉 no goals
    -/


theorem Set.WellFoundedOn.prod_lex_of_wellFoundedOn_fiber (hα : s.WellFoundedOn (rα on f))
    (hβ : ∀ a, (s ∩ f ⁻¹' {a}).WellFoundedOn (rβ on g)) :
    s.WellFoundedOn (Prod.Lex rα rβ on fun c => (f c, g c)) :=
  WellFounded.prod_lex_of_wellFoundedOn_fiber hα
    fun a ↦ ((hβ a).onFun (f := fun x => ⟨x, x.1.2, x.2⟩)).mono (fun _ _ h ↦ ‹_›)


/-- Stronger version of `PSigma.lex_wf`. Instead of requiring `rπ on g` to be well-founded, we only
require it to be well-founded on fibers of `f`. -/
theorem WellFounded.sigma_lex_of_wellFoundedOn_fiber (hι : WellFounded (rι on f))
    (hπ : ∀ i, (f ⁻¹' {i}).WellFoundedOn (rπ i on g i)) :
    WellFounded (Sigma.Lex rι rπ on fun c => ⟨f c, g (f c) c⟩) := by
  refine ((psigma_lex (wellFoundedOn_range.2 hι) fun a => hπ a).onFun
    (f := fun c => ⟨⟨_, c, rfl⟩, c, rfl⟩)).mono fun c c' h => ?_
  /-
    ι : Type u_1
    γ : Type u_4
    π : ι → Type u_5
    rι : ι → ι → Prop
    rπ : (i : ι) → π i → π i → Prop
    f : γ → ι
    g : (i : ι) → γ → π i
    hι : WellFounded (Function.onFun rι f)
    hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
    c c' : γ
    h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
    ⊢ Function.onFun (PSigma.Lex (fun a b => rι ↑a ↑b) fun a a_1 b => Function.onF …
  -/
  obtain h' | ⟨h', h''⟩ := Sigma.lex_iff.1 h
    /-
      case inl
      ι : Type u_1
      γ : Type u_4
      π : ι → Type u_5
      rι : ι → ι → Prop
      rπ : (i : ι) → π i → π i → Prop
      f : γ → ι
      g : (i : ι) → γ → π i
      hι : WellFounded (Function.onFun rι f)
      hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
      h' : rι ((fun c => ⟨f c, g (f c) c⟩) c).fst ((fun c => ⟨f c, g (f c) c⟩) c').fst
      ⊢ Function.onFun (PSigma.Lex (fun a b => rι ↑a ↑b) fun a a_1 b => Function.onF …
    -/
  · exact PSigma.Lex.left _ _ h'
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      ι : Type u_1
      γ : Type u_4
      π : ι → Type u_5
      rι : ι → ι → Prop
      rπ : (i : ι) → π i → π i → Prop
      f : γ → ι
      g : (i : ι) → γ → π i
      hι : WellFounded (Function.onFun rι f)
      hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
      h' : Eq ((fun c => ⟨f c, g (f c) c⟩) c).fst ((fun c => ⟨f c, g (f c) c⟩) c').fst
      h'' : rπ ((fun c => ⟨f c, g (f c) c⟩) c').fst (Eq.rec ((fun c => ⟨f c, g (f c) …
      ⊢ Function.onFun (PSigma.Lex (fun a b => rι ↑a ↑b) fun a a_1 b => Function.onF …
    -/
  · dsimp only [InvImage, (· on ·)] at h' ⊢
    /-
      case inr.intro
      ι : Type u_1
      γ : Type u_4
      π : ι → Type u_5
      rι : ι → ι → Prop
      rπ : (i : ι) → π i → π i → Prop
      f : γ → ι
      g : (i : ι) → γ → π i
      hι : WellFounded (Function.onFun rι f)
      hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
      c c' : γ
      h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
      h' : Eq (f c) (f c')
      h'' : rπ ((fun c => ⟨f c, g (f c) c⟩) c').fst (Eq.rec ((fun c => ⟨f c, g (f c) …
      ⊢ PSigma.Lex (fun a b => rι ↑a ↑b) (fun a a_1 b => rπ (↑a) (g ↑a ↑a_1) (g ↑a ↑ …
    -/
    convert PSigma.Lex.right (⟨_, c', rfl⟩ : range f) _ using 1; swap
      /-
        case inr.intro.convert_4
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        h' : Eq (f c) (f c')
        h'' : rπ ((fun c => ⟨f c, g (f c) c⟩) c').fst (Eq.rec ((fun c => ⟨f c, g (f c) …
        ⊢ ↑(Set.preimage f (Singleton.singleton ↑⟨f c', ⋯⟩))
      -/
    · exact ⟨c, h'⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        h' : Eq (f c) (f c')
        h'' : rπ ((fun c => ⟨f c, g (f c) c⟩) c').fst (Eq.rec ((fun c => ⟨f c, g (f c) …
        ⊢ Eq ⟨⟨f c, ⋯⟩, ⟨c, ⋯⟩⟩ ⟨⟨f c', ⋯⟩, ⟨c, h'⟩⟩
      -/
    · exact PSigma.subtype_ext (Subtype.ext h') rfl
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.convert_6
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        h' : Eq (f c) (f c')
        h'' : rπ ((fun c => ⟨f c, g (f c) c⟩) c').fst (Eq.rec ((fun c => ⟨f c, g (f c) …
        ⊢ rπ (↑⟨f c', ⋯⟩) (g ↑⟨f c', ⋯⟩ ↑⟨c, h'⟩) (g ↑⟨f c', ⋯⟩ ↑⟨c', ⋯⟩)
      -/
    · dsimp only [Subtype.coe_mk] at *
      /-
        case inr.intro.convert_6
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        h' : Eq (f c) (f c')
        h'' : rπ (f c') (Eq.rec (g (f c) c) h') (g (f c') c')
        ⊢ rπ (f c') (g (f c') c) (g (f c') c')
      -/
      revert h'
      /-
        case inr.intro.convert_6
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        ⊢ ∀ (h' : Eq (f c) (f c')), rπ (f c') (Eq.rec (g (f c) c) h') (g (f c') c') →  …
      -/
      generalize f c = d
      /-
        case inr.intro.convert_6
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        d : ι
        ⊢ ∀ (h' : Eq d (f c')), rπ (f c') (Eq.rec (g d c) h') (g (f c') c') → rπ (f c' …
      -/
      rintro rfl h''
      /-
        case inr.intro.convert_6
        ι : Type u_1
        γ : Type u_4
        π : ι → Type u_5
        rι : ι → ι → Prop
        rπ : (i : ι) → π i → π i → Prop
        f : γ → ι
        g : (i : ι) → γ → π i
        hι : WellFounded (Function.onFun rι f)
        hπ : ∀ (i : ι), (Set.preimage f (Singleton.singleton i)).WellFoundedOn (Functi …
        c c' : γ
        h : Function.onFun (Sigma.Lex rι rπ) (fun c => ⟨f c, g (f c) c⟩) c c'
        h'' : rπ (f c') (Eq.rec (g (f c') c) ⋯) (g (f c') c')
        ⊢ rπ (f c') (g (f c') c) (g (f c') c')
      -/
      exact h''
      /-
        🎉 no goals
      -/


theorem Set.WellFoundedOn.sigma_lex_of_wellFoundedOn_fiber (hι : s.WellFoundedOn (rι on f))
    (hπ : ∀ i, (s ∩ f ⁻¹' {i}).WellFoundedOn (rπ i on g i)) :
    s.WellFoundedOn (Sigma.Lex rι rπ on fun c => ⟨f c, g (f c) c⟩) := by
  /-
    ι : Type u_1
    γ : Type u_4
    π : ι → Type u_5
    rι : ι → ι → Prop
    rπ : (i : ι) → π i → π i → Prop
    f : γ → ι
    g : (i : ι) → γ → π i
    s : Set γ
    hι : s.WellFoundedOn (Function.onFun rι f)
    hπ : ∀ (i : ι), (Inter.inter s (Set.preimage f (Singleton.singleton i))).WellF …
    ⊢ s.WellFoundedOn (Function.onFun (Sigma.Lex rι rπ) fun c => ⟨f c, g (f c) c⟩)
  -/
  show WellFounded (Sigma.Lex rι rπ on fun c : s => ⟨f c, g (f c) c⟩)
  exact
    @WellFounded.sigma_lex_of_wellFoundedOn_fiber _ s _ _ rπ (fun c => f c) (fun i c => g _ c) hι
      fun i => ((hπ i).onFun (f := fun x => ⟨x, x.1.2, x.2⟩)).mono (fun b c h => ‹_›)



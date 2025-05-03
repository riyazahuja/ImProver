/-- The proposition that a function is eventually constant along a filter on the domain. -/
def EventuallyConst (f : α → β) (l : Filter α) : Prop := (map f l).Subsingleton


theorem HasBasis.eventuallyConst_iff {ι : Sort*} {p : ι → Prop} {s : ι → Set α}
    (h : l.HasBasis p s) : EventuallyConst f l ↔ ∃ i, p i ∧ ∀ x ∈ s i, ∀ y ∈ s i, f x = f y :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           l : Filter α
                                           f : α → β
                                           ι : Sort u_5
                                           p : ι → Prop
                                           s : ι → Set α
                                           h : l.HasBasis p s
                                           ⊢ Iff (Exists fun i => And (p i) (Set.image f (s i)).Subsingleton) (Exists fun …
                                         -/
  (h.map f).subsingleton_iff.trans <| by simp only [Set.Subsingleton, forall_mem_image]
                                         /-
                                           🎉 no goals
                                         -/


theorem HasBasis.eventuallyConst_iff' {ι : Sort*} {p : ι → Prop} {s : ι → Set α}
    {x : ι → α} (h : l.HasBasis p s) (hx : ∀ i, p i → x i ∈ s i) :
    EventuallyConst f l ↔ ∃ i, p i ∧ ∀ y ∈ s i, f y = f (x i) :=
  h.eventuallyConst_iff.trans <| exists_congr fun i ↦ and_congr_right fun hi ↦
    ⟨fun h ↦ (h · · (x i) (hx i hi)), fun h a ha b hb ↦ h a ha ▸ (h b hb).symm⟩


lemma eventuallyConst_iff_tendsto [Nonempty β] :
    EventuallyConst f l ↔ ∃ x, Tendsto f l (pure x) :=
  subsingleton_iff_exists_le_pure


alias ⟨EventuallyConst.exists_tendsto, _⟩ := eventuallyConst_iff_tendsto


theorem EventuallyConst.of_tendsto {x : β} (h : Tendsto f l (pure x)) : EventuallyConst f l :=
  have : Nonempty β := ⟨x⟩; eventuallyConst_iff_tendsto.2 ⟨x, h⟩


theorem eventuallyConst_iff_exists_eventuallyEq [Nonempty β] :
    EventuallyConst f l ↔ ∃ c, f =ᶠ[l] fun _ ↦ c :=
  subsingleton_iff_exists_singleton_mem


alias ⟨EventuallyConst.eventuallyEq_const, _⟩ := eventuallyConst_iff_exists_eventuallyEq


theorem eventuallyConst_pred' {p : α → Prop} :
    EventuallyConst p l ↔ (p =ᶠ[l] fun _ ↦ False) ∨ (p =ᶠ[l] fun _ ↦ True) := by
  /-
    α : Type u_1
    l : Filter α
    p : α → Prop
    ⊢ Iff (Filter.EventuallyConst p l) (Or (l.EventuallyEq p fun x => False) (l.Ev …
  -/
  simp only [eventuallyConst_iff_exists_eventuallyEq, Prop.exists_iff]
  /-
    🎉 no goals
  -/


theorem eventuallyConst_pred {p : α → Prop} :
    EventuallyConst p l ↔ (∀ᶠ x in l, p x) ∨ (∀ᶠ x in l, ¬p x) := by
  /-
    α : Type u_1
    l : Filter α
    p : α → Prop
    ⊢ Iff (Filter.EventuallyConst p l) (Or (Filter.Eventually (fun x => p x) l) (F …
  -/
  simp [eventuallyConst_pred', or_comm, EventuallyEq]
  /-
    🎉 no goals
  -/


theorem eventuallyConst_set' {s : Set α} :
    EventuallyConst s l ↔ (s =ᶠ[l] (∅ : Set α)) ∨ s =ᶠ[l] univ :=
  eventuallyConst_pred'


theorem eventuallyConst_set {s : Set α} :
    EventuallyConst s l ↔ (∀ᶠ x in l, x ∈ s) ∨ (∀ᶠ x in l, x ∉ s) :=
  eventuallyConst_pred


theorem eventuallyConst_preimage {s : Set β} {f : α → β} :
    EventuallyConst (f ⁻¹' s) l ↔ EventuallyConst s (map f l) :=
  .rfl


theorem EventuallyEq.eventuallyConst_iff {g : α → β} (h : f =ᶠ[l] g) :
    EventuallyConst f l ↔ EventuallyConst g l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    f g : α → β
    h : l.EventuallyEq f g
    ⊢ Iff (Filter.EventuallyConst f l) (Filter.EventuallyConst g l)
  -/
  simp only [EventuallyConst, map_congr h]
  /-
    🎉 no goals
  -/


@[simp] theorem eventuallyConst_id : EventuallyConst id l ↔ l.Subsingleton := Iff.rfl


@[simp] protected lemma bot : EventuallyConst f ⊥ := subsingleton_bot


@[simp]
protected lemma const (c : β) : EventuallyConst (fun _ ↦ c) l :=
  .of_tendsto tendsto_const_pure


protected lemma congr {g} (h : EventuallyConst f l) (hg : f =ᶠ[l] g) : EventuallyConst g l :=
  hg.eventuallyConst_iff.1 h


@[nontriviality]
lemma of_subsingleton_right [Subsingleton β] : EventuallyConst f l := .of_subsingleton


nonrec lemma anti {l'} (h : EventuallyConst f l) (hl' : l' ≤ l) : EventuallyConst f l' :=
  h.anti (map_mono hl')


@[nontriviality]
lemma of_subsingleton_left [Subsingleton α] : EventuallyConst f l :=
  .map .of_subsingleton f


lemma comp (h : EventuallyConst f l) (g : β → γ) : EventuallyConst (g ∘ f) l := h.map g


@[to_additive]
protected lemma inv [Inv β] (h : EventuallyConst f l) : EventuallyConst (f⁻¹) l := h.comp Inv.inv


lemma comp_tendsto {lb : Filter β} {g : β → γ} (hg : EventuallyConst g lb)
    (hf : Tendsto f l lb) : EventuallyConst (g ∘ f) l :=
  hg.anti hf


lemma apply {ι : Type*} {p : ι → Type*} {g : α → ∀ x, p x}
    (h : EventuallyConst g l) (i : ι) : EventuallyConst (g · i) l :=
  h.comp <| Function.eval i


lemma comp₂ {g : α → γ} (hf : EventuallyConst f l) (op : β → γ → δ) (hg : EventuallyConst g l) :
    EventuallyConst (fun x ↦ op (f x) (g x)) l :=
  ((hf.prod hg).map op.uncurry).anti <|
    (tendsto_map (f := op.uncurry)).comp (tendsto_map.prod_mk tendsto_map)


lemma prod_mk {g : α → γ} (hf : EventuallyConst f l) (hg : EventuallyConst g l) :
    EventuallyConst (fun x ↦ (f x, g x)) l :=
  hf.comp₂ Prod.mk hg


@[to_additive]
lemma mul [Mul β] {g : α → β} (hf : EventuallyConst f l) (hg : EventuallyConst g l) :
    EventuallyConst (f * g) l :=
  hf.comp₂ (· * ·) hg


@[to_additive]
lemma of_mulIndicator_const (h : EventuallyConst (s.mulIndicator fun _ ↦ c) l) (hc : c ≠ 1) :
    EventuallyConst s l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    inst✝ : One β
    s : Set α
    c : β
    h : Filter.EventuallyConst (s.mulIndicator fun x => c) l
    hc : Ne c 1
    ⊢ Filter.EventuallyConst s l
  -/
  simpa [Function.comp_def, hc, imp_false] using h.comp (· = c)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_const (h : EventuallyConst s l) (c : β) :
    EventuallyConst (s.mulIndicator fun _ ↦ c) l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    inst✝ : One β
    s : Set α
    h : Filter.EventuallyConst s l
    c : β
    ⊢ Filter.EventuallyConst (s.mulIndicator fun x => c) l
  -/
  classical exact h.comp (if · then c else 1)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_const_iff_of_ne (hc : c ≠ 1) :
    EventuallyConst (s.mulIndicator fun _ ↦ c) l ↔ EventuallyConst s l :=
  ⟨(of_mulIndicator_const · hc), (mulIndicator_const · c)⟩


@[to_additive (attr := simp)]
theorem mulIndicator_const_iff :
    EventuallyConst (s.mulIndicator fun _ ↦ c) l ↔ c = 1 ∨ EventuallyConst s l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    inst✝ : One β
    s : Set α
    c : β
    ⊢ Iff (Filter.EventuallyConst (s.mulIndicator fun x => c) l) (Or (Eq c 1) (Fil …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  rcases eq_or_ne c 1 with rfl | hc <;> simp [mulIndicator_const_iff_of_ne, *]
                                        /-
                                          🎉 no goals
                                        -/


lemma eventuallyConst_atTop [SemilatticeSup α] [Nonempty α] :
    EventuallyConst f atTop ↔ (∃ i, ∀ j, i ≤ j → f j = f i) :=
  (atTop_basis.eventuallyConst_iff' fun _ _ ↦ left_mem_Ici).trans <| by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : SemilatticeSup α
      inst✝ : Nonempty α
      ⊢ Iff (Exists fun i => And True (∀ (y : α), Membership.mem (Set.Ici i) y → Eq  …
    -/
    simp only [true_and, mem_Ici]
    /-
      🎉 no goals
    -/


lemma eventuallyConst_atTop_nat {f : ℕ → α} :
    EventuallyConst f atTop ↔ ∃ n, ∀ m, n ≤ m → f (m + 1) = f m := by
  /-
    α : Type u_1
    f : Nat → α
    ⊢ Iff (Filter.EventuallyConst f Filter.atTop) (Exists fun n => ∀ (m : Nat), LE …
  -/
  rw [eventuallyConst_atTop]
  /-
    α : Type u_1
    f : Nat → α
    ⊢ Iff (Exists fun i => ∀ (j : Nat), LE.le i j → Eq (f j) (f i)) (Exists fun n  …
  -/
  refine exists_congr fun n ↦ ⟨fun h m hm ↦ ?_, fun h m hm ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      f : Nat → α
      n : Nat
      h : ∀ (j : Nat), LE.le n j → Eq (f j) (f n)
      m : Nat
      hm : LE.le n m
      ⊢ Eq (f (HAdd.hAdd m 1)) (f m)
    -/
  · exact (h (m + 1) (hm.trans m.le_succ)).trans (h m hm).symm
    /-
      🎉 no goals
    -/
  · induction m, hm using Nat.le_induction with
    | base => rfl
    | succ m hm ihm => exact (h m hm).trans ihm



theorem formPerm_disjoint_iff (hl : Nodup l) (hl' : Nodup l') (hn : 2 ≤ l.length)
    (hn' : 2 ≤ l'.length) : Perm.Disjoint (formPerm l) (formPerm l') ↔ l.Disjoint l' := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hl : l.Nodup
    hl' : l'.Nodup
    hn : LE.le 2 l.length
    hn' : LE.le 2 l'.length
    ⊢ Iff (l.formPerm.Disjoint l'.formPerm) (l.Disjoint l')
  -/
  rw [disjoint_iff_eq_or_eq, List.Disjoint]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l l' : List α
    hl : l.Nodup
    hl' : l'.Nodup
    hn : LE.le 2 l.length
    hn' : LE.le 2 l'.length
    ⊢ Iff (∀ (x : α), Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)) (∀ ⦃a : α⦄, …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      ⊢ (∀ (x : α), Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)) → ∀ ⦃a : α⦄, Me …
    -/
  · rintro h x hx hx'
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      h : ∀ (x : α), Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
      x : α
      hx : Membership.mem l x
      hx' : Membership.mem l' x
      ⊢ False
    -/
    specialize h x
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      x : α
      hx : Membership.mem l x
      hx' : Membership.mem l' x
      h : Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
      ⊢ False
    -/
    rw [formPerm_apply_mem_eq_self_iff _ hl _ hx, formPerm_apply_mem_eq_self_iff _ hl' _ hx'] at h
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      x : α
      hx : Membership.mem l x
      hx' : Membership.mem l' x
      h : Or (LE.le l.length 1) (LE.le l'.length 1)
      ⊢ False
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      ⊢ (∀ ⦃a : α⦄, Membership.mem l a → Membership.mem l' a → False) → ∀ (x : α), O …
    -/
  · intro h x
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      h : ∀ ⦃a : α⦄, Membership.mem l a → Membership.mem l' a → False
      x : α
      ⊢ Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
    -/
    by_cases hx : x ∈ l
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      h : ∀ ⦃a : α⦄, Membership.mem l a → Membership.mem l' a → False
      x : α
      hx : Membership.mem l x
      ⊢ Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
    -/
    on_goal 1 => by_cases hx' : x ∈ l'
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        l l' : List α
        hl : l.Nodup
        hl' : l'.Nodup
        hn : LE.le 2 l.length
        hn' : LE.le 2 l'.length
        h : ∀ ⦃a : α⦄, Membership.mem l a → Membership.mem l' a → False
        x : α
        hx : Membership.mem l x
        hx' : Membership.mem l' x
        ⊢ Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
      -/
    · exact (h hx hx').elim
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      l l' : List α
      hl : l.Nodup
      hl' : l'.Nodup
      hn : LE.le 2 l.length
      hn' : LE.le 2 l'.length
      h : ∀ ⦃a : α⦄, Membership.mem l a → Membership.mem l' a → False
      x : α
      hx : Membership.mem l x
      hx' : Not (Membership.mem l' x)
      ⊢ Or (Eq (l.formPerm x) x) (Eq (l'.formPerm x) x)
    -/
    all_goals have := formPerm_eq_self_of_not_mem _ _ ‹_›; tauto
    /-
      🎉 no goals
    -/


theorem isCycle_formPerm (hl : Nodup l) (hn : 2 ≤ l.length) : IsCycle (formPerm l) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    hn : LE.le 2 l.length
    ⊢ l.formPerm.IsCycle
  -/
  cases' l with x l
    /-
      case nil
      α : Type u_1
      inst✝ : DecidableEq α
      hl : List.nil.Nodup
      hn : LE.le 2 List.nil.length
      ⊢ List.nil.formPerm.IsCycle
    -/
  · norm_num at hn
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    l : List α
    hl : (List.cons x l).Nodup
    hn : LE.le 2 (List.cons x l).length
    ⊢ (List.cons x l).formPerm.IsCycle
  -/
  induction' l with y l generalizing x
    /-
      case cons.nil
      α : Type u_1
      inst✝ : DecidableEq α
      x : α
      hl : (List.cons x List.nil).Nodup
      hn : LE.le 2 (List.cons x List.nil).length
      ⊢ (List.cons x List.nil).formPerm.IsCycle
    -/
  · norm_num at hn
    /-
      🎉 no goals
    -/
    /-
      case cons.cons
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      l : List α
      tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
      x : α
      hl : (List.cons x (List.cons y l)).Nodup
      hn : LE.le 2 (List.cons x (List.cons y l)).length
      ⊢ (List.cons x (List.cons y l)).formPerm.IsCycle
    -/
  · use x
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      y : α
      l : List α
      tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
      x : α
      hl : (List.cons x (List.cons y l)).Nodup
      hn : LE.le 2 (List.cons x (List.cons y l)).length
      ⊢ And (Ne ((List.cons x (List.cons y l)).formPerm x) x) (∀ ⦃y_1 : α⦄, Ne ((Lis …
    -/
    constructor
      /-
        case h.left
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        ⊢ Ne ((List.cons x (List.cons y l)).formPerm x) x
      -/
    · rwa [formPerm_apply_mem_ne_self_iff _ hl _ (mem_cons_self _ _)]
      /-
        🎉 no goals
      -/
      /-
        case h.right
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        ⊢ ∀ ⦃y_1 : α⦄, Ne ((List.cons x (List.cons y l)).formPerm y_1) y_1 → (List.con …
      -/
    · intro w hw
      /-
        case h.right
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        w : α
        hw : Ne ((List.cons x (List.cons y l)).formPerm w) w
        ⊢ (List.cons x (List.cons y l)).formPerm.SameCycle x w
      -/
      have : w ∈ x::y::l := mem_of_formPerm_ne_self _ _ hw
      /-
        case h.right
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        w : α
        hw : Ne ((List.cons x (List.cons y l)).formPerm w) w
        this : Membership.mem (List.cons x (List.cons y l)) w
        ⊢ (List.cons x (List.cons y l)).formPerm.SameCycle x w
      -/
      obtain ⟨k, hk, rfl⟩ := getElem_of_mem this
      /-
        case h.right.intro.intro
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        k : Nat
        hk : LT.lt k (List.cons x (List.cons y l)).length
        hw : Ne ((List.cons x (List.cons y l)).formPerm (GetElem.getElem (List.cons x  …
        this : Membership.mem (List.cons x (List.cons y l)) (GetElem.getElem (List.con …
        ⊢ (List.cons x (List.cons y l)).formPerm.SameCycle x (GetElem.getElem (List.co …
      -/
      use k
      /-
        case h
        α : Type u_1
        inst✝ : DecidableEq α
        y : α
        l : List α
        tail_ih✝ : ∀ (x : α), (List.cons x l).Nodup → LE.le 2 (List.cons x l).length → …
        x : α
        hl : (List.cons x (List.cons y l)).Nodup
        hn : LE.le 2 (List.cons x (List.cons y l)).length
        k : Nat
        hk : LT.lt k (List.cons x (List.cons y l)).length
        hw : Ne ((List.cons x (List.cons y l)).formPerm (GetElem.getElem (List.cons x  …
        this : Membership.mem (List.cons x (List.cons y l)) (GetElem.getElem (List.con …
        ⊢ Eq ((HPow.hPow (List.cons x (List.cons y l)).formPerm ↑k) x) (GetElem.getEle …
      -/
      simp only [zpow_natCast, formPerm_pow_apply_head _ _ hl k, Nat.mod_eq_of_lt hk]
      /-
        🎉 no goals
      -/


theorem pairwise_sameCycle_formPerm (hl : Nodup l) (hn : 2 ≤ l.length) :
    Pairwise l.formPerm.SameCycle l :=
  Pairwise.imp_mem.mpr
    (pairwise_of_forall fun _ _ hx hy =>
      (isCycle_formPerm hl hn).sameCycle ((formPerm_apply_mem_ne_self_iff _ hl _ hx).mpr hn)
        ((formPerm_apply_mem_ne_self_iff _ hl _ hy).mpr hn))


theorem cycleOf_formPerm (hl : Nodup l) (hn : 2 ≤ l.length) (x) :
    cycleOf l.attach.formPerm x = l.attach.formPerm :=
                                      /-
                                        α : Type u_1
                                        inst✝ : DecidableEq α
                                        l : List α
                                        hl : l.Nodup
                                        hn : LE.le 2 l.length
                                        x : Subtype fun x => Membership.mem l x
                                        ⊢ LE.le 2 l.attach.length
                                      -/
  have hn : 2 ≤ l.attach.length := by rwa [← length_attach] at hn
                                      /-
                                        🎉 no goals
                                      -/
                                 /-
                                   α : Type u_1
                                   inst✝ : DecidableEq α
                                   l : List α
                                   hl : l.Nodup
                                   hn✝ : LE.le 2 l.length
                                   x : Subtype fun x => Membership.mem l x
                                   hn : LE.le 2 l.attach.length
                                   ⊢ l.attach.Nodup
                                 -/
  have hl : l.attach.Nodup := by rwa [← nodup_attach] at hl
                                 /-
                                   🎉 no goals
                                 -/
  (isCycle_formPerm hl hn).cycleOf_eq
    ((formPerm_apply_mem_ne_self_iff _ hl _ (mem_attach _ _)).mpr hn)


theorem cycleType_formPerm (hl : Nodup l) (hn : 2 ≤ l.length) :
    cycleType l.attach.formPerm = {l.length} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    hn : LE.le 2 l.length
    ⊢ Eq l.attach.formPerm.cycleType (Singleton.singleton l.length)
  -/
  rw [← length_attach] at hn
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    hn : LE.le 2 l.attach.length
    ⊢ Eq l.attach.formPerm.cycleType (Singleton.singleton l.length)
  -/
  rw [← nodup_attach] at hl
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.attach.Nodup
    hn : LE.le 2 l.attach.length
    ⊢ Eq l.attach.formPerm.cycleType (Singleton.singleton l.length)
  -/
  rw [cycleType_eq [l.attach.formPerm]]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.attach.Nodup
      hn : LE.le 2 l.attach.length
      ⊢ Eq (↑(List.map (Function.comp Finset.card Equiv.Perm.support) (List.cons l.a …
    -/
  · simp only [map, Function.comp_apply]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.attach.Nodup
      hn : LE.le 2 l.attach.length
      ⊢ Eq (↑(List.cons l.attach.formPerm.support.card List.nil)) (Singleton.singlet …
    -/
    rw [support_formPerm_of_nodup _ hl, card_toFinset, dedup_eq_self.mpr hl]
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        hl : l.attach.Nodup
        hn : LE.le 2 l.attach.length
        ⊢ Eq (↑(List.cons l.attach.length List.nil)) (Singleton.singleton l.length)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        hl : l.attach.Nodup
        hn : LE.le 2 l.attach.length
        ⊢ ∀ (x : Subtype fun x => Membership.mem l x), Ne l.attach (List.cons x List.n …
      -/
    · intro x h
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l : List α
        hl : l.attach.Nodup
        hn : LE.le 2 l.attach.length
        x : Subtype fun x => Membership.mem l x
        h : Eq l.attach (List.cons x List.nil)
        ⊢ False
      -/
      simp [h, Nat.succ_le_succ_iff] at hn
      /-
        🎉 no goals
      -/
    /-
      case h0
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.attach.Nodup
      hn : LE.le 2 l.attach.length
      ⊢ Eq (List.cons l.attach.formPerm List.nil).prod l.attach.formPerm
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h1
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.attach.Nodup
      hn : LE.le 2 l.attach.length
      ⊢ ∀ (σ : Equiv.Perm (Subtype fun x => Membership.mem l x)), Membership.mem (Li …
    -/
  · simpa using isCycle_formPerm hl hn
    /-
      🎉 no goals
    -/
    /-
      case h2
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      hl : l.attach.Nodup
      hn : LE.le 2 l.attach.length
      ⊢ List.Pairwise Equiv.Perm.Disjoint (List.cons l.attach.formPerm List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem formPerm_apply_mem_eq_next (hl : Nodup l) (x : α) (hx : x ∈ l) :
    formPerm l x = next l x hx := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ Eq (l.formPerm x) (l.next x hx)
  -/
  obtain ⟨k, rfl⟩ := get_of_mem hx
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    ⊢ Eq (l.formPerm (l.get k)) (l.next (l.get k) hx)
  -/
  rw [next_get _ hl, get_eq_getElem, formPerm_apply_getElem _ hl]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- A cycle `s : Cycle α`, given `Nodup s` can be interpreted as an `Equiv.Perm α`
where each element in the list is permuted to the next one, defined as `formPerm`.
-/
def formPerm : ∀ s : Cycle α, Nodup s → Equiv.Perm α :=
  fun s => Quotient.hrecOn s (fun l _ => List.formPerm l) fun l₁ l₂ (h : l₁ ~r l₂) => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s✝ s : Cycle α
      l₁ l₂ : List α
      h : l₁.IsRotated l₂
      ⊢ HEq (fun x => l₁.formPerm) fun x => l₂.formPerm
    -/
    apply Function.hfunext
      /-
        case hα
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ s : Cycle α
        l₁ l₂ : List α
        h : l₁.IsRotated l₂
        ⊢ Eq (Cycle.Nodup (Quotient.mk (List.IsRotated.setoid α) l₁)) (Cycle.Nodup (Qu …
      -/
    · ext
      /-
        case hα.a
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ s : Cycle α
        l₁ l₂ : List α
        h : l₁.IsRotated l₂
        ⊢ Iff (Cycle.Nodup (Quotient.mk (List.IsRotated.setoid α) l₁)) (Cycle.Nodup (Q …
      -/
      exact h.nodup_iff
      /-
        🎉 no goals
      -/
      /-
        case h
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ s : Cycle α
        l₁ l₂ : List α
        h : l₁.IsRotated l₂
        ⊢ ∀ (a : Cycle.Nodup (Quotient.mk (List.IsRotated.setoid α) l₁)) (a' : Cycle.N …
      -/
    · intro h₁ h₂ _
      /-
        case h
        α : Type u_1
        inst✝ : DecidableEq α
        s✝ s : Cycle α
        l₁ l₂ : List α
        h : l₁.IsRotated l₂
        h₁ : Cycle.Nodup (Quotient.mk (List.IsRotated.setoid α) l₁)
        h₂ : Cycle.Nodup (Quotient.mk (List.IsRotated.setoid α) l₂)
        a✝ : HEq h₁ h₂
        ⊢ HEq l₁.formPerm l₂.formPerm
      -/
      exact heq_of_eq (formPerm_eq_of_isRotated h₁ h)
      /-
        🎉 no goals
      -/


@[simp]
theorem formPerm_coe (l : List α) (hl : l.Nodup) : formPerm (l : Cycle α) hl = l.formPerm :=
  rfl


theorem formPerm_subsingleton (s : Cycle α) (h : Subsingleton s) : formPerm s h.nodup = 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    h : s.Subsingleton
    ⊢ Eq (s.formPerm ⋯) 1
  -/
  induction' s using Quot.inductionOn with s
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    ⊢ Eq (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) s) ⋯) 1
  -/
  simp only [formPerm_coe, mk_eq_coe]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    h : Cycle.Subsingleton (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    ⊢ Eq s.formPerm 1
  -/
  simp only [length_subsingleton_iff, length_coe, mk_eq_coe] at h
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : List α
    h : LE.le s.length 1
    ⊢ Eq s.formPerm 1
  -/
  cases' s with hd tl
    /-
      case h.nil
      α : Type u_1
      inst✝ : DecidableEq α
      h : LE.le List.nil.length 1
      ⊢ Eq List.nil.formPerm 1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      α : Type u_1
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      h : LE.le (List.cons hd tl).length 1
      ⊢ Eq (List.cons hd tl).formPerm 1
    -/
  · simp only [length_eq_zero, add_le_iff_nonpos_left, List.length, nonpos_iff_eq_zero] at h
    /-
      case h.cons
      α : Type u_1
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      h : Eq tl List.nil
      ⊢ Eq (List.cons hd tl).formPerm 1
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem isCycle_formPerm (s : Cycle α) (h : Nodup s) (hn : Nontrivial s) :
    IsCycle (formPerm s h) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    h : s.Nodup
    hn : s.Nontrivial
    ⊢ (s.formPerm h).IsCycle
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a✝ : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    hn : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    ⊢ (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) h).IsCycle
  -/
  exact List.isCycle_formPerm h (length_nontrivial hn)
  /-
    🎉 no goals
  -/


theorem support_formPerm [Fintype α] (s : Cycle α) (h : Nodup s) (hn : Nontrivial s) :
    support (formPerm s h) = s.toFinset := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Cycle α
    h : s.Nodup
    hn : s.Nontrivial
    ⊢ Eq (s.formPerm h).support s.toFinset
  -/
  induction' s using Quot.inductionOn with s
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    hn : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    ⊢ Eq (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) s) h).support (Cycl …
  -/
  refine support_formPerm_of_nodup s h ?_
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    hn : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
    ⊢ ∀ (x : α), Ne s (List.cons x List.nil)
  -/
  rintro _ rfl
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x✝ : α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons x✝ List.nil))
    hn : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons x✝ List …
    ⊢ False
  -/
  simpa [Nat.succ_le_succ_iff] using length_nontrivial hn
  /-
    🎉 no goals
  -/


theorem formPerm_eq_self_of_not_mem (s : Cycle α) (h : Nodup s) (x : α) (hx : x ∉ s) :
    formPerm s h x = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    h : s.Nodup
    x : α
    hx : Not (Membership.mem s x)
    ⊢ Eq ((s.formPerm h) x) x
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    a✝ : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    hx : Not (Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) x)
    ⊢ Eq ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) h) x) x
  -/
  simpa using List.formPerm_eq_self_of_not_mem _ _ hx
  /-
    🎉 no goals
  -/


theorem formPerm_apply_mem_eq_next (s : Cycle α) (h : Nodup s) (x : α) (hx : x ∈ s) :
    formPerm s h x = next s h x hx := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    h : s.Nodup
    x : α
    hx : Membership.mem s x
    ⊢ Eq ((s.formPerm h) x) (s.next h x hx)
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    x : α
    a✝ : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) x
    ⊢ Eq ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) a✝) h) x) (Cycle.n …
  -/
  simpa using List.formPerm_apply_mem_eq_next h _ (by simp_all)
  /-
    🎉 no goals
  -/


nonrec theorem formPerm_reverse (s : Cycle α) (h : Nodup s) :
    formPerm s.reverse (nodup_reverse_iff.mpr h) = (formPerm s h)⁻¹ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Cycle α
    h : s.Nodup
    ⊢ Eq (s.reverse.formPerm ⋯) (Inv.inv (s.formPerm h))
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a✝ : List α
    h : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)
    ⊢ Eq ((Cycle.reverse (Quot.mk (⇑(List.IsRotated.setoid α)) a✝)).formPerm ⋯) (I …
  -/
  simpa using formPerm_reverse _
  /-
    🎉 no goals
  -/


nonrec theorem formPerm_eq_formPerm_iff {α : Type*} [DecidableEq α] {s s' : Cycle α} {hs : s.Nodup}
    {hs' : s'.Nodup} :
    s.formPerm hs = s'.formPerm hs' ↔ s = s' ∨ s.Subsingleton ∧ s'.Subsingleton := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s s' : Cycle α
    hs : s.Nodup
    hs' : s'.Nodup
    ⊢ Iff (Eq (s.formPerm hs) (s'.formPerm hs')) (Or (Eq s s') (And s.Subsingleton …
  -/
  rw [Cycle.length_subsingleton_iff, Cycle.length_subsingleton_iff]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s s' : Cycle α
    hs : s.Nodup
    hs' : s'.Nodup
    ⊢ Iff (Eq (s.formPerm hs) (s'.formPerm hs')) (Or (Eq s s') (And (LE.le s.lengt …
  -/
  revert s s'
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    ⊢ ∀ {s s' : Cycle α} {hs : s.Nodup} {hs' : s'.Nodup}, Iff (Eq (s.formPerm hs)  …
  -/
  intro s s'
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s s' : Cycle α
    ⊢ ∀ {hs : s.Nodup} {hs' : s'.Nodup}, Iff (Eq (s.formPerm hs) (s'.formPerm hs') …
  -/
  apply @Quotient.inductionOn₂' _ _ _ _ _ s s'
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s s' : Cycle α
    ⊢ ∀ (a₁ a₂ : List α) {hs : Cycle.Nodup (Quotient.mk'' a₁)} {hs' : Cycle.Nodup  …
  -/
  intro l l' hl hl'
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s s' : Cycle α
    l l' : List α
    hl : Cycle.Nodup (Quotient.mk'' l)
    hl' : Cycle.Nodup (Quotient.mk'' l')
    ⊢ Iff (Eq (Cycle.formPerm (Quotient.mk'' l) hl) (Cycle.formPerm (Quotient.mk'' …
  -/
  simpa using formPerm_eq_formPerm_iff hl hl'
  /-
    🎉 no goals
  -/


/-- `Equiv.Perm.toList (f : Perm α) (x : α)` generates the list `[x, f x, f (f x), ...]`
until looping. That means when `f x = x`, `toList f x = []`.
-/
def toList : List α :=
  (List.range (cycleOf p x).support.card).map fun k => (p ^ k) x


@[simp]
                                                      /-
                                                        α : Type u_1
                                                        inst✝¹ : Fintype α
                                                        inst✝ : DecidableEq α
                                                        x : α
                                                        ⊢ Eq (Equiv.Perm.toList 1 x) List.nil
                                                      -/
theorem toList_one : toList (1 : Perm α) x = [] := by simp [toList, cycleOf_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                                                   /-
                                                                                     α : Type u_1
                                                                                     inst✝¹ : Fintype α
                                                                                     inst✝ : DecidableEq α
                                                                                     p : Equiv.Perm α
                                                                                     x : α
                                                                                     ⊢ Iff (Eq (p.toList x) List.nil) (Not (Membership.mem p.support x))
                                                                                   -/
theorem toList_eq_nil_iff {p : Perm α} {x} : toList p x = [] ↔ x ∉ p.support := by simp [toList]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝¹ : Fintype α
                                                                                 inst✝ : DecidableEq α
                                                                                 p : Equiv.Perm α
                                                                                 x : α
                                                                                 ⊢ Eq (p.toList x).length (p.cycleOf x).support.card
                                                                               -/
theorem length_toList : length (toList p x) = (cycleOf p x).support.card := by simp [toList]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem toList_ne_singleton (y : α) : toList p x ≠ [y] := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    ⊢ Ne (p.toList x) (List.cons y List.nil)
  -/
  intro H
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    H : Eq (p.toList x) (List.cons y List.nil)
    ⊢ False
  -/
  simpa [card_support_ne_one] using congr_arg length H
  /-
    🎉 no goals
  -/


theorem two_le_length_toList_iff_mem_support {p : Perm α} {x : α} :
                                                  /-
                                                    α : Type u_1
                                                    inst✝¹ : Fintype α
                                                    inst✝ : DecidableEq α
                                                    p : Equiv.Perm α
                                                    x : α
                                                    ⊢ Iff (LE.le 2 (p.toList x).length) (Membership.mem p.support x)
                                                  -/
    2 ≤ length (toList p x) ↔ x ∈ p.support := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem length_toList_pos_of_mem_support (h : x ∈ p.support) : 0 < length (toList p x) :=
  zero_lt_two.trans_le (two_le_length_toList_iff_mem_support.mpr h)


theorem get_toList (n : ℕ) (hn : n < length (toList p x)) :
                                               /-
                                                 α : Type u_1
                                                 inst✝¹ : Fintype α
                                                 inst✝ : DecidableEq α
                                                 p : Equiv.Perm α
                                                 x : α
                                                 n : Nat
                                                 hn : LT.lt n (p.toList x).length
                                                 ⊢ Eq ((p.toList x).get ⟨n, hn⟩) ((HPow.hPow p n) x)
                                               -/
    (toList p x).get ⟨n, hn⟩ = (p ^ n) x := by simp [toList]
                                               /-
                                                 🎉 no goals
                                               -/


theorem toList_get_zero (h : x ∈ p.support) :
                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝¹ : Fintype α
                                                                               inst✝ : DecidableEq α
                                                                               p : Equiv.Perm α
                                                                               x : α
                                                                               h : Membership.mem p.support x
                                                                               ⊢ Eq ((p.toList x).get ⟨0, ⋯⟩) x
                                                                             -/
    (toList p x).get ⟨0, (length_toList_pos_of_mem_support _ _ h)⟩ = x := by simp [toList]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem mem_toList_iff {y : α} : y ∈ toList p x ↔ SameCycle p x y ∧ x ∈ p.support := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    ⊢ Iff (Membership.mem (p.toList x) y) (And (p.SameCycle x y) (Membership.mem p …
  -/
  simp only [toList, mem_range, mem_map]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    ⊢ Iff (Exists fun a => And (LT.lt a (p.cycleOf x).support.card) (Eq ((HPow.hPo …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x y : α
      ⊢ (Exists fun a => And (LT.lt a (p.cycleOf x).support.card) (Eq ((HPow.hPow p  …
    -/
  · rintro ⟨n, hx, rfl⟩
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      n : Nat
      hx : LT.lt n (p.cycleOf x).support.card
      ⊢ And (p.SameCycle x ((HPow.hPow p n) x)) (Membership.mem p.support x)
    -/
    refine ⟨⟨n, rfl⟩, ?_⟩
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      n : Nat
      hx : LT.lt n (p.cycleOf x).support.card
      ⊢ Membership.mem p.support x
    -/
    contrapose! hx
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      n : Nat
      hx : Not (Membership.mem p.support x)
      ⊢ LE.le (p.cycleOf x).support.card n
    -/
    rw [← support_cycleOf_eq_nil_iff] at hx
    /-
      case mp.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      n : Nat
      hx : Eq (p.cycleOf x).support EmptyCollection.emptyCollection
      ⊢ LE.le (p.cycleOf x).support.card n
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x y : α
      ⊢ And (p.SameCycle x y) (Membership.mem p.support x) → Exists fun a => And (LT …
    -/
  · rintro ⟨h, hx⟩
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x y : α
      h : p.SameCycle x y
      hx : Membership.mem p.support x
      ⊢ Exists fun a => And (LT.lt a (p.cycleOf x).support.card) (Eq ((HPow.hPow p a …
    -/
    simpa using h.exists_pow_eq_of_mem_support hx
    /-
      🎉 no goals
    -/


theorem nodup_toList (p : Perm α) (x : α) : Nodup (toList p x) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    ⊢ (p.toList x).Nodup
  -/
  by_cases hx : p x = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Eq (p x) x
      ⊢ (p.toList x).Nodup
    -/
  · rw [← not_mem_support, ← toList_eq_nil_iff] at hx
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Eq (p.toList x) List.nil
      ⊢ (p.toList x).Nodup
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Not (Eq (p x) x)
    ⊢ (p.toList x).Nodup
  -/
  have hc : IsCycle (cycleOf p x) := isCycle_cycleOf p hx
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Not (Eq (p x) x)
    hc : (p.cycleOf x).IsCycle
    ⊢ (p.toList x).Nodup
  -/
  rw [nodup_iff_injective_get]
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Not (Eq (p x) x)
    hc : (p.cycleOf x).IsCycle
    ⊢ Function.Injective (p.toList x).get
  -/
  intro ⟨n, hn⟩ ⟨m, hm⟩
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Not (Eq (p x) x)
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn : LT.lt n (p.toList x).length
    m : Nat
    hm : LT.lt m (p.toList x).length
    ⊢ Eq ((p.toList x).get ⟨n, hn⟩) ((p.toList x).get ⟨m, hm⟩) → Eq ⟨n, hn⟩ ⟨m, hm⟩
  -/
  rw [length_toList, ← hc.orderOf] at hm hn
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Not (Eq (p x) x)
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt n (p.toList x).length
    hn : LT.lt n (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt m (p.toList x).length
    hm : LT.lt m (orderOf (p.cycleOf x))
    ⊢ Eq ((p.toList x).get ⟨n, hn✝⟩) ((p.toList x).get ⟨m, hm✝⟩) → Eq ⟨n, hn✝⟩ ⟨m, …
  -/
  rw [← cycleOf_apply_self, ← Ne, ← mem_support] at hx
  rw [get_toList, get_toList, ← cycleOf_pow_apply_self p x n, ←
    cycleOf_pow_apply_self p x m]
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt n (p.toList x).length
    hn : LT.lt n (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt m (p.toList x).length
    hm : LT.lt m (orderOf (p.cycleOf x))
    ⊢ Eq ((HPow.hPow (p.cycleOf x) n) x) ((HPow.hPow (p.cycleOf x) m) x) → Eq ⟨n,  …
  -/
  cases' n with n <;> cases' m with m
    /-
      case neg.zero.zero
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Membership.mem (p.cycleOf x).support x
      hc : (p.cycleOf x).IsCycle
      hn✝ : LT.lt 0 (p.toList x).length
      hn : LT.lt 0 (orderOf (p.cycleOf x))
      hm✝ : LT.lt 0 (p.toList x).length
      hm : LT.lt 0 (orderOf (p.cycleOf x))
      ⊢ Eq ((HPow.hPow (p.cycleOf x) 0) x) ((HPow.hPow (p.cycleOf x) 0) x) → Eq ⟨0,  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [← hc.support_pow_of_pos_of_lt_orderOf m.zero_lt_succ hm, mem_support,
      cycleOf_pow_apply_self] at hx
    /-
      case neg.zero.succ
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hc : (p.cycleOf x).IsCycle
      hn✝ : LT.lt 0 (p.toList x).length
      hn : LT.lt 0 (orderOf (p.cycleOf x))
      m : Nat
      hx : Ne ((HPow.hPow p m.succ) x) x
      hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
      hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
      ⊢ Eq ((HPow.hPow (p.cycleOf x) 0) x) ((HPow.hPow (p.cycleOf x) (HAdd.hAdd m 1) …
    -/
    simp [hx.symm]
    /-
      🎉 no goals
    -/
  · rw [← hc.support_pow_of_pos_of_lt_orderOf n.zero_lt_succ hn, mem_support,
      cycleOf_pow_apply_self] at hx
    /-
      case neg.succ.zero
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hc : (p.cycleOf x).IsCycle
      n : Nat
      hx : Ne ((HPow.hPow p n.succ) x) x
      hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
      hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
      hm✝ : LT.lt 0 (p.toList x).length
      hm : LT.lt 0 (orderOf (p.cycleOf x))
      ⊢ Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) 0 …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    ⊢ Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) ( …
  -/
  intro h
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
    ⊢ Eq ⟨HAdd.hAdd n 1, hn✝⟩ ⟨HAdd.hAdd m 1, hm✝⟩
  -/
  have hn' : ¬orderOf (p.cycleOf x) ∣ n.succ := Nat.not_dvd_of_pos_of_lt n.zero_lt_succ hn
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
    hn' : Not (Dvd.dvd (orderOf (p.cycleOf x)) n.succ)
    ⊢ Eq ⟨HAdd.hAdd n 1, hn✝⟩ ⟨HAdd.hAdd m 1, hm✝⟩
  -/
  have hm' : ¬orderOf (p.cycleOf x) ∣ m.succ := Nat.not_dvd_of_pos_of_lt m.zero_lt_succ hm
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
    hn' : Not (Dvd.dvd (orderOf (p.cycleOf x)) n.succ)
    hm' : Not (Dvd.dvd (orderOf (p.cycleOf x)) m.succ)
    ⊢ Eq ⟨HAdd.hAdd n 1, hn✝⟩ ⟨HAdd.hAdd m 1, hm✝⟩
  -/
  rw [← hc.support_pow_eq_iff] at hn' hm'
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
    hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
    hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
    ⊢ Eq ⟨HAdd.hAdd n 1, hn✝⟩ ⟨HAdd.hAdd m 1, hm✝⟩
  -/
  rw [Fin.mk_eq_mk, ← Nat.mod_eq_of_lt hn, ← Nat.mod_eq_of_lt hm, ← pow_inj_mod]
  /-
    case neg.succ.succ
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    hx : Membership.mem (p.cycleOf x).support x
    hc : (p.cycleOf x).IsCycle
    n : Nat
    hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
    hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
    m : Nat
    hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
    hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
    h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
    hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
    hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
    ⊢ Eq (HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) (HPow.hPow (p.cycleOf x) (HAdd. …
  -/
  refine support_congr ?_ ?_
    /-
      case neg.succ.succ.refine_1
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Membership.mem (p.cycleOf x).support x
      hc : (p.cycleOf x).IsCycle
      n : Nat
      hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
      hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
      m : Nat
      hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
      hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
      h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
      hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
      hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
      ⊢ HasSubset.Subset (HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)).support (HPow.hPo …
    -/
  · rw [hm', hn']
    /-
      🎉 no goals
    -/
    /-
      case neg.succ.succ.refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Membership.mem (p.cycleOf x).support x
      hc : (p.cycleOf x).IsCycle
      n : Nat
      hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
      hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
      m : Nat
      hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
      hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
      h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
      hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
      hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
      ⊢ ∀ (x_1 : α), Membership.mem (HPow.hPow (p.cycleOf x) (HAdd.hAdd m 1)).suppor …
    -/
  · rw [hm']
    /-
      case neg.succ.succ.refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Membership.mem (p.cycleOf x).support x
      hc : (p.cycleOf x).IsCycle
      n : Nat
      hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
      hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
      m : Nat
      hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
      hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
      h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
      hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
      hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
      ⊢ ∀ (x_1 : α), Membership.mem (p.cycleOf x).support x_1 → Eq ((HPow.hPow (p.cy …
    -/
    intro y hy
    /-
      case neg.succ.succ.refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      hx : Membership.mem (p.cycleOf x).support x
      hc : (p.cycleOf x).IsCycle
      n : Nat
      hn✝ : LT.lt (HAdd.hAdd n 1) (p.toList x).length
      hn : LT.lt (HAdd.hAdd n 1) (orderOf (p.cycleOf x))
      m : Nat
      hm✝ : LT.lt (HAdd.hAdd m 1) (p.toList x).length
      hm : LT.lt (HAdd.hAdd m 1) (orderOf (p.cycleOf x))
      h : Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) x) ((HPow.hPow (p.cycleOf x) …
      hn' : Eq (HPow.hPow (p.cycleOf x) n.succ).support (p.cycleOf x).support
      hm' : Eq (HPow.hPow (p.cycleOf x) m.succ).support (p.cycleOf x).support
      y : α
      hy : Membership.mem (p.cycleOf x).support y
      ⊢ Eq ((HPow.hPow (p.cycleOf x) (HAdd.hAdd n 1)) y) ((HPow.hPow (p.cycleOf x) ( …
    -/
    obtain ⟨k, rfl⟩ := hc.exists_pow_eq (mem_support.mp hx) (mem_support.mp hy)
    rw [← mul_apply, (Commute.pow_pow_self _ _ _).eq, mul_apply, h, ← mul_apply, ← mul_apply,
      (Commute.pow_pow_self _ _ _).eq]


theorem next_toList_eq_apply (p : Perm α) (x y : α) (hy : y ∈ toList p x) :
    next (toList p x) y hy = p y := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy : Membership.mem (p.toList x) y
    ⊢ Eq ((p.toList x).next y hy) (p y)
  -/
  rw [mem_toList_iff] at hy
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    ⊢ Eq ((p.toList x).next y hy✝) (p y)
  -/
  obtain ⟨k, hk, hk'⟩ := hy.left.exists_pow_eq_of_mem_support hy.right
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((HPow.hPow p k) x) y
    ⊢ Eq ((p.toList x).next y hy✝) (p y)
  -/
  rw [← get_toList p x k (by simpa using hk)] at hk'
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((p.toList x).get ⟨k, ⋯⟩) y
    ⊢ Eq ((p.toList x).next y hy✝) (p y)
  -/
  simp_rw [← hk']
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((p.toList x).get ⟨k, ⋯⟩) y
    ⊢ Eq ((p.toList x).next ((p.toList x).get ⟨k, ⋯⟩) ⋯) (p ((p.toList x).get ⟨k,  …
  -/
  rw [next_get _ (nodup_toList _ _), get_toList, get_toList, ← mul_apply, ← pow_succ']
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((p.toList x).get ⟨k, ⋯⟩) y
    ⊢ Eq ((HPow.hPow p (HMod.hMod (HAdd.hAdd (↑⟨k, ⋯⟩) 1) (p.toList x).length)) x) …
  -/
  simp_rw [length_toList]
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((p.toList x).get ⟨k, ⋯⟩) y
    ⊢ Eq ((HPow.hPow p (HMod.hMod (HAdd.hAdd k 1) (p.cycleOf x).support.card)) x)  …
  -/
  rw [← pow_mod_orderOf_cycleOf_apply p (k + 1), IsCycle.orderOf]
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x y : α
    hy✝ : Membership.mem (p.toList x) y
    hy : And (p.SameCycle x y) (Membership.mem p.support x)
    k : Nat
    hk : LT.lt k (p.cycleOf x).support.card
    hk' : Eq ((p.toList x).get ⟨k, ⋯⟩) y
    ⊢ (p.cycleOf x).IsCycle
  -/
  exact isCycle_cycleOf _ (mem_support.mp hy.right)
  /-
    🎉 no goals
  -/


theorem toList_pow_apply_eq_rotate (p : Perm α) (x : α) (k : ℕ) :
    p.toList ((p ^ k) x) = (p.toList x).rotate k := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    k : Nat
    ⊢ Eq (p.toList ((HPow.hPow p k) x)) ((p.toList x).rotate k)
  -/
  apply ext_get
    /-
      case hl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      k : Nat
      ⊢ Eq (p.toList ((HPow.hPow p k) x)).length ((p.toList x).rotate k).length
    -/
  · simp only [length_toList, cycleOf_self_apply_pow, length_rotate]
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      k : Nat
      ⊢ ∀ (n : Nat) (h₁ : LT.lt n (p.toList ((HPow.hPow p k) x)).length) (h₂ : LT.lt …
    -/
  · intro n hn hn'
    rw [get_toList, get_rotate, get_toList, length_toList,
      pow_mod_card_support_cycleOf_self_apply, pow_add, mul_apply]


theorem SameCycle.toList_isRotated {f : Perm α} {x y : α} (h : SameCycle f x y) :
    toList f x ~r toList f y := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x y : α
    h : f.SameCycle x y
    ⊢ (f.toList x).IsRotated (f.toList y)
  -/
  by_cases hx : x ∈ f.support
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      h : f.SameCycle x y
      hx : Membership.mem f.support x
      ⊢ (f.toList x).IsRotated (f.toList y)
    -/
  · obtain ⟨_ | k, _, hy⟩ := h.exists_pow_eq_of_mem_support hx
      /-
        case pos.intro.zero.intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : Equiv.Perm α
        x y : α
        h : f.SameCycle x y
        hx : Membership.mem f.support x
        left✝ : LT.lt 0 (f.cycleOf x).support.card
        hy : Eq ((HPow.hPow f 0) x) y
        ⊢ (f.toList x).IsRotated (f.toList y)
      -/
    · simp only [coe_one, id, pow_zero] at hy
      -- Porting note: added `IsRotated.refl`
      /-
        case pos.intro.zero.intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : Equiv.Perm α
        x y : α
        h : f.SameCycle x y
        hx : Membership.mem f.support x
        left✝ : LT.lt 0 (f.cycleOf x).support.card
        hy : Eq x y
        ⊢ (f.toList x).IsRotated (f.toList y)
      -/
      simp [hy, IsRotated.refl]
      /-
        🎉 no goals
      -/
    /-
      case pos.intro.succ.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      h : f.SameCycle x y
      hx : Membership.mem f.support x
      k : Nat
      left✝ : LT.lt (HAdd.hAdd k 1) (f.cycleOf x).support.card
      hy : Eq ((HPow.hPow f (HAdd.hAdd k 1)) x) y
      ⊢ (f.toList x).IsRotated (f.toList y)
    -/
    use k.succ
    /-
      case h
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      h : f.SameCycle x y
      hx : Membership.mem f.support x
      k : Nat
      left✝ : LT.lt (HAdd.hAdd k 1) (f.cycleOf x).support.card
      hy : Eq ((HPow.hPow f (HAdd.hAdd k 1)) x) y
      ⊢ Eq ((f.toList x).rotate k.succ) (f.toList y)
    -/
    rw [← toList_pow_apply_eq_rotate, hy]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      h : f.SameCycle x y
      hx : Not (Membership.mem f.support x)
      ⊢ (f.toList x).IsRotated (f.toList y)
    -/
  · rw [toList_eq_nil_iff.mpr hx, isRotated_nil_iff', eq_comm, toList_eq_nil_iff]
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      h : f.SameCycle x y
      hx : Not (Membership.mem f.support x)
      ⊢ Not (Membership.mem f.support y)
    -/
    rwa [← h.mem_support_iff]
    /-
      🎉 no goals
    -/


theorem pow_apply_mem_toList_iff_mem_support {n : ℕ} : (p ^ n) x ∈ p.toList x ↔ x ∈ p.support := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    n : Nat
    ⊢ Iff (Membership.mem (p.toList x) ((HPow.hPow p n) x)) (Membership.mem p.supp …
  -/
  rw [mem_toList_iff, and_iff_right_iff_imp]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    n : Nat
    ⊢ Membership.mem p.support x → p.SameCycle x ((HPow.hPow p n) x)
  -/
  refine fun _ => SameCycle.symm ?_
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    p : Equiv.Perm α
    x : α
    n : Nat
    x✝ : Membership.mem p.support x
    ⊢ p.SameCycle ((HPow.hPow p n) x) x
  -/
  rw [sameCycle_pow_left]
  /-
    🎉 no goals
  -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     inst✝¹ : Fintype α
                                                                                     inst✝ : DecidableEq α
                                                                                     x : α
                                                                                     ⊢ Eq (List.nil.formPerm.toList x) List.nil
                                                                                   -/
theorem toList_formPerm_nil (x : α) : toList (formPerm ([] : List α)) x = [] := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


                                                                                 /-
                                                                                   α : Type u_1
                                                                                   inst✝¹ : Fintype α
                                                                                   inst✝ : DecidableEq α
                                                                                   x y : α
                                                                                   ⊢ Eq ((List.cons x List.nil).formPerm.toList y) List.nil
                                                                                 -/
theorem toList_formPerm_singleton (x y : α) : toList (formPerm [x]) y = [] := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem toList_formPerm_nontrivial (l : List α) (hl : 2 ≤ l.length) (hn : Nodup l) :
    toList (formPerm l) (l.get ⟨0, (zero_lt_two.trans_le hl)⟩) = l := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    ⊢ Eq (l.formPerm.toList (l.get ⟨0, ⋯⟩)) l
  -/
  have hc : l.formPerm.IsCycle := List.isCycle_formPerm hn hl
  have hs : l.formPerm.support = l.toFinset := by
    refine support_formPerm_of_nodup _ hn ?_
    rintro _ rfl
    simp [Nat.succ_le_succ_iff] at hl
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    hc : l.formPerm.IsCycle
    hs : Eq l.formPerm.support l.toFinset
    ⊢ Eq (l.formPerm.toList (l.get ⟨0, ⋯⟩)) l
  -/
  rw [toList, hc.cycleOf_eq (mem_support.mp _), hs, card_toFinset, dedup_eq_self.mpr hn]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List α
      hl : LE.le 2 l.length
      hn : l.Nodup
      hc : l.formPerm.IsCycle
      hs : Eq l.formPerm.support l.toFinset
      ⊢ Eq (List.map (fun k => (HPow.hPow l.formPerm k) (l.get ⟨0, ⋯⟩)) (List.range  …
    -/
  · refine ext_getElem (by simp) fun k hk hk' => ?_
    simp only [get_eq_getElem, formPerm_pow_apply_getElem _ hn, zero_add, getElem_map,
      getElem_range, Nat.mod_eq_of_lt hk']
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List α
      hl : LE.le 2 l.length
      hn : l.Nodup
      hc : l.formPerm.IsCycle
      hs : Eq l.formPerm.support l.toFinset
      ⊢ Membership.mem l.formPerm.support (l.get ⟨0, ⋯⟩)
    -/
  · simp [hs]
    /-
      🎉 no goals
    -/


theorem toList_formPerm_isRotated_self (l : List α) (hl : 2 ≤ l.length) (hn : Nodup l) (x : α)
    (hx : x ∈ l) : toList (formPerm l) x ~r l := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    x : α
    hx : Membership.mem l x
    ⊢ (l.formPerm.toList x).IsRotated l
  -/
  obtain ⟨k, hk, rfl⟩ := get_of_mem hx
  /-
    case intro.refl
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    ⊢ (l.formPerm.toList (l.get k)).IsRotated l
  -/
  have hr : l ~r l.rotate k := ⟨k, rfl⟩
  /-
    case intro.refl
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    hr : l.IsRotated (l.rotate ↑k)
    ⊢ (l.formPerm.toList (l.get k)).IsRotated l
  -/
  rw [formPerm_eq_of_isRotated hn hr]
  /-
    case intro.refl
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    hr : l.IsRotated (l.rotate ↑k)
    ⊢ ((l.rotate ↑k).formPerm.toList (l.get k)).IsRotated l
  -/
  rw [get_eq_get_rotate l k k]
  /-
    case intro.refl
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    hr : l.IsRotated (l.rotate ↑k)
    ⊢ ((l.rotate ↑k).formPerm.toList ((l.rotate ↑k).get ⟨HMod.hMod (HAdd.hAdd (HSu …
  -/
  simp only [Nat.mod_eq_of_lt k.2, tsub_add_cancel_of_le (le_of_lt k.2), Nat.mod_self]
  /-
    case intro.refl
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List α
    hl : LE.le 2 l.length
    hn : l.Nodup
    k : Fin l.length
    hx : Membership.mem l (l.get k)
    hr : l.IsRotated (l.rotate ↑k)
    ⊢ ((l.rotate ↑k).formPerm.toList ((l.rotate ↑k).get ⟨0, ⋯⟩)).IsRotated l
  -/
  rw [toList_formPerm_nontrivial]
    /-
      case intro.refl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List α
      hl : LE.le 2 l.length
      hn : l.Nodup
      k : Fin l.length
      hx : Membership.mem l (l.get k)
      hr : l.IsRotated (l.rotate ↑k)
      ⊢ (l.rotate ↑k).IsRotated l
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.refl.hl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List α
      hl : LE.le 2 l.length
      hn : l.Nodup
      k : Fin l.length
      hx : Membership.mem l (l.get k)
      hr : l.IsRotated (l.rotate ↑k)
      ⊢ LE.le 2 (l.rotate ↑k).length
    -/
  · simpa using hl
    /-
      🎉 no goals
    -/
    /-
      case intro.refl.hn
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      l : List α
      hl : LE.le 2 l.length
      hn : l.Nodup
      k : Fin l.length
      hx : Membership.mem l (l.get k)
      hr : l.IsRotated (l.rotate ↑k)
      ⊢ (l.rotate ↑k).Nodup
    -/
  · simpa using hn
    /-
      🎉 no goals
    -/


theorem formPerm_toList (f : Perm α) (x : α) : formPerm (toList f x) = f.cycleOf x := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x : α
    ⊢ Eq (f.toList x).formPerm (f.cycleOf x)
  -/
  by_cases hx : f x = x
  · rw [(cycleOf_eq_one_iff f).mpr hx, toList_eq_nil_iff.mpr (not_mem_support.mpr hx),
      formPerm_nil]
  /-
    case neg
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x : α
    hx : Not (Eq (f x) x)
    ⊢ Eq (f.toList x).formPerm (f.cycleOf x)
  -/
  ext y
  /-
    case neg.H
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x : α
    hx : Not (Eq (f x) x)
    y : α
    ⊢ Eq ((f.toList x).formPerm y) ((f.cycleOf x) y)
  -/
  by_cases hy : SameCycle f x y
    /-
      case pos
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      y : α
      hy : f.SameCycle x y
      ⊢ Eq ((f.toList x).formPerm y) ((f.cycleOf x) y)
    -/
  · obtain ⟨k, _, rfl⟩ := hy.exists_pow_eq_of_mem_support (mem_support.mpr hx)
    rw [cycleOf_apply_apply_pow_self, List.formPerm_apply_mem_eq_next (nodup_toList f x),
      next_toList_eq_apply, pow_succ', mul_apply]
    /-
      case pos.intro.intro.hy
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      k : Nat
      left✝ : LT.lt k (f.cycleOf x).support.card
      hy : f.SameCycle x ((HPow.hPow f k) x)
      ⊢ Membership.mem (f.toList x) ((HPow.hPow f k) x)
    -/
    rw [mem_toList_iff]
    /-
      case pos.intro.intro.hy
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      k : Nat
      left✝ : LT.lt k (f.cycleOf x).support.card
      hy : f.SameCycle x ((HPow.hPow f k) x)
      ⊢ And (f.SameCycle x ((HPow.hPow f k) x)) (Membership.mem f.support x)
    -/
    exact ⟨⟨k, rfl⟩, mem_support.mpr hx⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      y : α
      hy : Not (f.SameCycle x y)
      ⊢ Eq ((f.toList x).formPerm y) ((f.cycleOf x) y)
    -/
  · rw [cycleOf_apply_of_not_sameCycle hy, formPerm_apply_of_not_mem]
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x : α
      hx : Not (Eq (f x) x)
      y : α
      hy : Not (f.SameCycle x y)
      ⊢ Not (Membership.mem (f.toList x) y)
    -/
    simp [mem_toList_iff, hy]
    /-
      🎉 no goals
    -/


/-- Given a cyclic `f : Perm α`, generate the `Cycle α` in the order
of application of `f`. Implemented by finding an element `x : α`
in the support of `f` in `Finset.univ`, and iterating on using
`Equiv.Perm.toList f x`.
-/
def toCycle (f : Perm α) (hf : IsCycle f) : Cycle α :=
  Multiset.recOn (Finset.univ : Finset α).val (Quot.mk _ [])
    (fun x _ l => if f x = x then l else toList f x)
    (by
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x : α
        f : Equiv.Perm α
        hf : f.IsCycle
        ⊢ ∀ (a a' : α), Multiset α → ∀ (b : Cycle α), HEq (ite (Eq (f a) a) (ite (Eq ( …
      -/
      intro x y _ s
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        f : Equiv.Perm α
        hf : f.IsCycle
        x y : α
        m✝ : Multiset α
        s : Cycle α
        ⊢ HEq (ite (Eq (f x) x) (ite (Eq (f y) y) s ↑(f.toList y)) ↑(f.toList x)) (ite …
      -/
      refine heq_of_eq ?_
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        f : Equiv.Perm α
        hf : f.IsCycle
        x y : α
        m✝ : Multiset α
        s : Cycle α
        ⊢ Eq (ite (Eq (f x) x) (ite (Eq (f y) y) s ↑(f.toList y)) ↑(f.toList x)) (ite  …
      -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
      split_ifs with hx hy hy <;> try rfl
      /-
        case neg
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        f : Equiv.Perm α
        hf : f.IsCycle
        x y : α
        m✝ : Multiset α
        s : Cycle α
        hx : Not (Eq (f x) x)
        hy : Not (Eq (f y) y)
        ⊢ Eq ↑(f.toList x) ↑(f.toList y)
      -/
      have hc : SameCycle f x y := IsCycle.sameCycle hf hx hy
      /-
        case neg
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        f : Equiv.Perm α
        hf : f.IsCycle
        x y : α
        m✝ : Multiset α
        s : Cycle α
        hx : Not (Eq (f x) x)
        hy : Not (Eq (f y) y)
        hc : f.SameCycle x y
        ⊢ Eq ↑(f.toList x) ↑(f.toList y)
      -/
      exact Quotient.sound' hc.toList_isRotated)
      /-
        🎉 no goals
      -/


theorem toCycle_eq_toList (f : Perm α) (hf : IsCycle f) (x : α) (hx : f x ≠ x) :
    toCycle f hf = toList f x := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    x : α
    hx : Ne (f x) x
    ⊢ Eq (f.toCycle hf) ↑(f.toList x)
  -/
  have key : (Finset.univ : Finset α).val = x ::ₘ Finset.univ.val.erase x := by simp
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    x : α
    hx : Ne (f x) x
    key : Eq Finset.univ.val (Multiset.cons x (Finset.univ.val.erase x))
    ⊢ Eq (f.toCycle hf) ↑(f.toList x)
  -/
  rw [toCycle, key]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    x : α
    hx : Ne (f x) x
    key : Eq Finset.univ.val (Multiset.cons x (Finset.univ.val.erase x))
    ⊢ Eq ((Multiset.cons x (Finset.univ.val.erase x)).recOn (Quot.mk (⇑(List.IsRot …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


theorem nodup_toCycle (f : Perm α) (hf : IsCycle f) : (toCycle f hf).Nodup := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    ⊢ (f.toCycle hf).Nodup
  -/
  obtain ⟨x, hx, -⟩ := id hf
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    x : α
    hx : Ne (f x) x
    ⊢ (f.toCycle hf).Nodup
  -/
  simpa [toCycle_eq_toList f hf x hx] using nodup_toList _ _
  /-
    🎉 no goals
  -/


theorem nontrivial_toCycle (f : Perm α) (hf : IsCycle f) : (toCycle f hf).Nontrivial := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    ⊢ (f.toCycle hf).Nontrivial
  -/
  obtain ⟨x, hx, -⟩ := id hf
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    x : α
    hx : Ne (f x) x
    ⊢ (f.toCycle hf).Nontrivial
  -/
  simp [toCycle_eq_toList f hf x hx, hx, Cycle.nontrivial_coe_nodup_iff (nodup_toList _ _)]
  /-
    🎉 no goals
  -/


/-- Any cyclic `f : Perm α` is isomorphic to the nontrivial `Cycle α`
that corresponds to repeated application of `f`.
The forward direction is implemented by `Equiv.Perm.toCycle`.
-/
def isoCycle : { f : Perm α // IsCycle f } ≃ { s : Cycle α // s.Nodup ∧ s.Nontrivial } where
  toFun f := ⟨toCycle (f : Perm α) f.prop, nodup_toCycle (f : Perm α) f.prop,
    nontrivial_toCycle _ f.prop⟩
  invFun s := ⟨(s : Cycle α).formPerm s.prop.left, (s : Cycle α).isCycle_formPerm _ s.prop.right⟩
  left_inv f := by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      f : Subtype fun f => f.IsCycle
      ⊢ Eq ((fun s => ⟨(↑s).formPerm ⋯, ⋯⟩) ((fun f => ⟨(↑f).toCycle ⋯, ⋯⟩) f)) f
    -/
    obtain ⟨x, hx, -⟩ := id f.prop
    simpa [toCycle_eq_toList (f : Perm α) f.prop x hx, formPerm_toList, Subtype.ext_iff] using
      f.prop.cycleOf_eq hx
  right_inv s := by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      s : Subtype fun s => And s.Nodup s.Nontrivial
      ⊢ Eq ((fun f => ⟨(↑f).toCycle ⋯, ⋯⟩) ((fun s => ⟨(↑s).formPerm ⋯, ⋯⟩) s)) s
    -/
    rcases s with ⟨⟨s⟩, hn, ht⟩
    /-
      case mk.mk.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x : α
      val✝ : Cycle α
      s : List α
      hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ⊢ Eq ((fun f => ⟨(↑f).toCycle ⋯, ⋯⟩) ((fun s => ⟨(↑s).formPerm ⋯, ⋯⟩) ⟨Quot.mk …
    -/
    obtain ⟨x, -, -, hx, -⟩ := id ht
    /-
      case mk.mk.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x✝ : α
      val✝ : Cycle α
      s : List α
      hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      x : α
      hx : Membership.mem (Quot.mk (⇑(List.IsRotated.setoid α)) s) x
      ⊢ Eq ((fun f => ⟨(↑f).toCycle ⋯, ⋯⟩) ((fun s => ⟨(↑s).formPerm ⋯, ⋯⟩) ⟨Quot.mk …
    -/
    have hl : 2 ≤ s.length := by simpa using Cycle.length_nontrivial ht
    simp only [Cycle.mk_eq_coe, Cycle.nodup_coe_iff, Cycle.mem_coe_iff, Subtype.coe_mk,
      Cycle.formPerm_coe] at hn hx ⊢
    /-
      case mk.mk.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x✝ : α
      val✝ : Cycle α
      s : List α
      hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      x : α
      hl : LE.le 2 s.length
      hn : s.Nodup
      hx : Membership.mem s x
      ⊢ Eq ⟨s.formPerm.toCycle ⋯, ⋯⟩ ⟨↑s, ⋯⟩
    -/
    apply Subtype.ext
    /-
      case mk.mk.intro.intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x✝ : α
      val✝ : Cycle α
      s : List α
      hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      x : α
      hl : LE.le 2 s.length
      hn : s.Nodup
      hx : Membership.mem s x
      ⊢ Eq ↑⟨s.formPerm.toCycle ⋯, ⋯⟩ ↑⟨↑s, ⋯⟩
    -/
    dsimp
    /-
      case mk.mk.intro.intro.intro.intro.intro.a
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      p : Equiv.Perm α
      x✝ : α
      val✝ : Cycle α
      s : List α
      hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
      x : α
      hl : LE.le 2 s.length
      hn : s.Nodup
      hx : Membership.mem s x
      ⊢ Eq (s.formPerm.toCycle ⋯) ↑s
    -/
    rw [toCycle_eq_toList _ _ x]
      /-
        case mk.mk.intro.intro.intro.intro.intro.a
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        val✝ : Cycle α
        s : List α
        hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        x : α
        hl : LE.le 2 s.length
        hn : s.Nodup
        hx : Membership.mem s x
        ⊢ Eq ↑(s.formPerm.toList x) ↑s
      -/
    · refine Quotient.sound' ?_
      /-
        case mk.mk.intro.intro.intro.intro.intro.a
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        val✝ : Cycle α
        s : List α
        hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        x : α
        hl : LE.le 2 s.length
        hn : s.Nodup
        hx : Membership.mem s x
        ⊢ (List.IsRotated.setoid α) (s.formPerm.toList x) s
      -/
      exact toList_formPerm_isRotated_self _ hl hn _ hx
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.intro.intro.intro.intro.intro.a
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        p : Equiv.Perm α
        x✝ : α
        val✝ : Cycle α
        s : List α
        hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
        x : α
        hl : LE.le 2 s.length
        hn : s.Nodup
        hx : Membership.mem s x
        ⊢ Ne (s.formPerm x) x
      -/
    · rw [← mem_support, support_formPerm_of_nodup _ hn]
        /-
          case mk.mk.intro.intro.intro.intro.intro.a
          α : Type u_1
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          p : Equiv.Perm α
          x✝ : α
          val✝ : Cycle α
          s : List α
          hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
          ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
          x : α
          hl : LE.le 2 s.length
          hn : s.Nodup
          hx : Membership.mem s x
          ⊢ Membership.mem s.toFinset x
        -/
      · simpa using hx
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.intro.intro.intro.intro.intro.a
          α : Type u_1
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          p : Equiv.Perm α
          x✝ : α
          val✝ : Cycle α
          s : List α
          hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) s)
          ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) s)
          x : α
          hl : LE.le 2 s.length
          hn : s.Nodup
          hx : Membership.mem s x
          ⊢ ∀ (x : α), Ne s (List.cons x List.nil)
        -/
      · rintro _ rfl
        /-
          case mk.mk.intro.intro.intro.intro.intro.a
          α : Type u_1
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          p : Equiv.Perm α
          x✝¹ : α
          val✝ : Cycle α
          x x✝ : α
          hn✝ : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons x✝ List.nil))
          ht : Cycle.Nontrivial (Quot.mk (⇑(List.IsRotated.setoid α)) (List.cons x✝ List …
          hl : LE.le 2 (List.cons x✝ List.nil).length
          hn : (List.cons x✝ List.nil).Nodup
          hx : Membership.mem (List.cons x✝ List.nil) x
          ⊢ False
        -/
        simp [Nat.succ_le_succ_iff] at hl
        /-
          🎉 no goals
        -/


theorem IsCycle.existsUnique_cycle {f : Perm α} (hf : IsCycle f) :
    ∃! s : Cycle α, ∃ h : s.Nodup, s.formPerm h = f := by
  /-
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    ⊢ ExistsUnique fun s => Exists fun h => Eq (s.formPerm h) f
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    val✝ : Fintype α
    ⊢ ExistsUnique fun s => Exists fun h => Eq (s.formPerm h) f
  -/
  obtain ⟨x, hx, hy⟩ := id hf
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    val✝ : Fintype α
    x : α
    hx : Ne (f x) x
    hy : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
    ⊢ ExistsUnique fun s => Exists fun h => Eq (s.formPerm h) f
  -/
  refine ⟨f.toList x, ⟨nodup_toList f x, ?_⟩, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      val✝ : Fintype α
      x : α
      hx : Ne (f x) x
      hy : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
      ⊢ Eq ((↑(f.toList x)).formPerm ⋯) f
    -/
  · simp [formPerm_toList, hf.cycleOf_eq hx]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      val✝ : Fintype α
      x : α
      hx : Ne (f x) x
      hy : ∀ ⦃y : α⦄, Ne (f y) y → f.SameCycle x y
      ⊢ ∀ (y : Cycle α), (fun s => Exists fun h => Eq (s.formPerm h) f) y → Eq y ↑(f …
    -/
  · rintro ⟨l⟩ ⟨hn, rfl⟩
    /-
      case intro.intro.intro.refine_2.mk.intro
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      val✝ : Fintype α
      x : α
      y✝ : Cycle α
      l : List α
      hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
      hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
      hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
      hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
      ⊢ Eq (Quot.mk (⇑(List.IsRotated.setoid α)) l) ↑((Cycle.formPerm (Quot.mk (⇑(Li …
    -/
    simp only [Cycle.mk_eq_coe, Cycle.coe_eq_coe, Subtype.coe_mk, Cycle.formPerm_coe]
    /-
      case intro.intro.intro.refine_2.mk.intro
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      val✝ : Fintype α
      x : α
      y✝ : Cycle α
      l : List α
      hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
      hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
      hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
      hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
      ⊢ l.IsRotated (l.formPerm.toList x)
    -/
    refine (toList_formPerm_isRotated_self _ ?_ hn _ ?_).symm
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_1
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        ⊢ LE.le 2 l.length
      -/
    · contrapose! hx
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_1
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        hx : LT.lt l.length 2
        ⊢ Eq ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
      -/
      suffices formPerm l = 1 by simp [this]
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_1
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        hx : LT.lt l.length 2
        ⊢ Eq l.formPerm 1
      -/
      rw [formPerm_eq_one_iff _ hn]
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_1
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        hx : LT.lt l.length 2
        ⊢ LE.le l.length 1
      -/
      exact Nat.le_of_lt_succ hx
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_2
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        ⊢ Membership.mem l x
      -/
    · rw [← mem_toFinset]
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_2
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        ⊢ Membership.mem l.toFinset x
      -/
      refine support_formPerm_le l ?_
      /-
        case intro.intro.intro.refine_2.mk.intro.refine_2
        α : Type u_1
        inst✝¹ : Finite α
        inst✝ : DecidableEq α
        val✝ : Fintype α
        x : α
        y✝ : Cycle α
        l : List α
        hn : Cycle.Nodup (Quot.mk (⇑(List.IsRotated.setoid α)) l)
        hf : (Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn).IsCycle
        hx : Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) hn) x) x
        hy : ∀ ⦃y : α⦄, Ne ((Cycle.formPerm (Quot.mk (⇑(List.IsRotated.setoid α)) l) h …
        ⊢ Membership.mem l.formPerm.support x
      -/
      simpa using hx
      /-
        🎉 no goals
      -/


theorem IsCycle.existsUnique_cycle_subtype {f : Perm α} (hf : IsCycle f) :
    ∃! s : { s : Cycle α // s.Nodup }, (s : Cycle α).formPerm s.prop = f := by
  /-
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    ⊢ ExistsUnique fun s => Eq ((↑s).formPerm ⋯) f
  -/
  obtain ⟨s, ⟨hs, rfl⟩, hs'⟩ := hf.existsUnique_cycle
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    hf : (s.formPerm hs).IsCycle
    hs' : ∀ (y : Cycle α), (fun s_1 => Exists fun h => Eq (s_1.formPerm h) (s.form …
    ⊢ ExistsUnique fun s_1 => Eq ((↑s_1).formPerm ⋯) (s.formPerm hs)
  -/
  refine ⟨⟨s, hs⟩, rfl, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    hf : (s.formPerm hs).IsCycle
    hs' : ∀ (y : Cycle α), (fun s_1 => Exists fun h => Eq (s_1.formPerm h) (s.form …
    ⊢ ∀ (y : Subtype fun s => s.Nodup), (fun s_1 => Eq ((↑s_1).formPerm ⋯) (s.form …
  -/
  rintro ⟨t, ht⟩ ht'
  /-
    case intro.intro.intro.mk
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    s : Cycle α
    hs : s.Nodup
    hf : (s.formPerm hs).IsCycle
    hs' : ∀ (y : Cycle α), (fun s_1 => Exists fun h => Eq (s_1.formPerm h) (s.form …
    t : Cycle α
    ht : t.Nodup
    ht' : Eq ((↑⟨t, ht⟩).formPerm ⋯) (s.formPerm hs)
    ⊢ Eq ⟨t, ht⟩ ⟨s, hs⟩
  -/
  simpa using hs' _ ⟨ht, ht'⟩
  /-
    🎉 no goals
  -/


theorem IsCycle.existsUnique_cycle_nontrivial_subtype {f : Perm α} (hf : IsCycle f) :
    ∃! s : { s : Cycle α // s.Nodup ∧ s.Nontrivial }, (s : Cycle α).formPerm s.prop.left = f := by
  /-
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    ⊢ ExistsUnique fun s => Eq ((↑s).formPerm ⋯) f
  -/
  obtain ⟨⟨s, hn⟩, hs, hs'⟩ := hf.existsUnique_cycle_subtype
  /-
    case intro.mk.intro
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hf : f.IsCycle
    s : Cycle α
    hn : s.Nodup
    hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
    hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
    ⊢ ExistsUnique fun s => Eq ((↑s).formPerm ⋯) f
  -/
  refine ⟨⟨s, hn, ?_⟩, ?_, ?_⟩
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      s : Cycle α
      hn : s.Nodup
      hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
      ⊢ s.Nontrivial
    -/
  · rw [hn.nontrivial_iff]
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      s : Cycle α
      hn : s.Nodup
      hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
      ⊢ Not s.Subsingleton
    -/
    subst f
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      s : Cycle α
      hn : s.Nodup
      hf : ((↑⟨s, hn⟩).formPerm ⋯).IsCycle
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s_1 => Eq ((↑s_1).formPerm ⋯) ((↑ …
      ⊢ Not s.Subsingleton
    -/
    intro H
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      s : Cycle α
      hn : s.Nodup
      hf : ((↑⟨s, hn⟩).formPerm ⋯).IsCycle
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s_1 => Eq ((↑s_1).formPerm ⋯) ((↑ …
      H : s.Subsingleton
      ⊢ False
    -/
    refine hf.ne_one ?_
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      s : Cycle α
      hn : s.Nodup
      hf : ((↑⟨s, hn⟩).formPerm ⋯).IsCycle
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s_1 => Eq ((↑s_1).formPerm ⋯) ((↑ …
      H : s.Subsingleton
      ⊢ Eq ((↑⟨s, hn⟩).formPerm ⋯) 1
    -/
    simpa using Cycle.formPerm_subsingleton _ H
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.refine_2
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      s : Cycle α
      hn : s.Nodup
      hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
      ⊢ (fun s => Eq ((↑s).formPerm ⋯) f) ⟨s, ⋯⟩
    -/
  · simpa using hs
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.refine_3
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      s : Cycle α
      hn : s.Nodup
      hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
      ⊢ ∀ (y : Subtype fun s => And s.Nodup s.Nontrivial), (fun s => Eq ((↑s).formPe …
    -/
  · rintro ⟨t, ht, ht'⟩ ht''
    /-
      case intro.mk.intro.refine_3.mk.intro
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      hf : f.IsCycle
      s : Cycle α
      hn : s.Nodup
      hs : Eq ((↑⟨s, hn⟩).formPerm ⋯) f
      hs' : ∀ (y : Subtype fun s => s.Nodup), (fun s => Eq ((↑s).formPerm ⋯) f) y →  …
      t : Cycle α
      ht : t.Nodup
      ht' : t.Nontrivial
      ht'' : Eq ((↑⟨t, ⋯⟩).formPerm ⋯) f
      ⊢ Eq ⟨t, ⋯⟩ ⟨s, ⋯⟩
    -/
    simpa using hs' ⟨t, ht⟩ ht''
    /-
      🎉 no goals
    -/


/-- Any cyclic `f : Perm α` is isomorphic to the nontrivial `Cycle α`
that corresponds to repeated application of `f`.
The forward direction is implemented by finding this `Cycle α` using `Fintype.choose`.
-/
def isoCycle' : { f : Perm α // IsCycle f } ≃ { s : Cycle α // s.Nodup ∧ s.Nontrivial } :=
  let f : { s : Cycle α // s.Nodup ∧ s.Nontrivial } → { f : Perm α // IsCycle f } :=
    fun s => ⟨(s : Cycle α).formPerm s.prop.left, (s : Cycle α).isCycle_formPerm _ s.prop.right⟩
  { toFun := Fintype.bijInv (show Function.Bijective f by
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : (Subtype fun s => And s.Nodup s.Nontrivial) → Subtype fun f => f.IsCycle : …
        ⊢ Function.Bijective f
      -/
      rw [Function.bijective_iff_existsUnique]
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f : (Subtype fun s => And s.Nodup s.Nontrivial) → Subtype fun f => f.IsCycle : …
        ⊢ ∀ (b : Subtype fun f => f.IsCycle), ExistsUnique fun a => Eq (f a) b
      -/
      rintro ⟨f, hf⟩
      /-
        case mk
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f✝ : (Subtype fun s => And s.Nodup s.Nontrivial) → Subtype fun f => f.IsCycle  …
        f : Equiv.Perm α
        hf : f.IsCycle
        ⊢ ExistsUnique fun a => Eq (f✝ a) ⟨f, hf⟩
      -/
      simp only [Subtype.ext_iff]
      /-
        case mk
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        f✝ : (Subtype fun s => And s.Nodup s.Nontrivial) → Subtype fun f => f.IsCycle  …
        f : Equiv.Perm α
        hf : f.IsCycle
        ⊢ ExistsUnique fun a => Eq ((↑a).formPerm ⋯) f
      -/
      exact hf.existsUnique_cycle_nontrivial_subtype)
      /-
        🎉 no goals
      -/
    invFun := f
    left_inv := Fintype.rightInverse_bijInv _
    right_inv := Fintype.leftInverse_bijInv _ }

-- mutes `'decide' tactic does nothing [linter.unusedTactic]`

set_option linter.unusedTactic false in
notation3 (prettyPrint := false) "c["(l", "* => foldr (h t => List.cons h t) List.nil)"]" =>
  Cycle.formPerm (Cycle.ofList l) (Iff.mpr Cycle.nodup_coe_iff (by decide))


unsafe instance repr_perm [Repr α] : Repr (Perm α) :=
  ⟨fun f _ => repr (Multiset.pmap (fun (g : Perm α) (hg : g.IsCycle) => isoCycle ⟨g, hg⟩)
    (Perm.cycleFactorsFinset f).val -- toCycle is faster?
    fun _ hg => (mem_cycleFactorsFinset_iff.mp (Finset.mem_def.mpr hg)).left)⟩



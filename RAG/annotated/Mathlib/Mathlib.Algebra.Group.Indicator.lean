/-- `Set.mulIndicator s f a` is `f a` if `a ∈ s`, `1` otherwise. -/
@[to_additive "`Set.indicator s f a` is `f a` if `a ∈ s`, `0` otherwise."]
noncomputable def mulIndicator (s : Set α) (f : α → M) (x : α) : M :=
  haveI := Classical.decPred (· ∈ s)
  if x ∈ s then f x else 1


@[to_additive (attr := simp)]
theorem piecewise_eq_mulIndicator [DecidablePred (· ∈ s)] : s.piecewise f 1 = s.mulIndicator f :=
  funext fun _ => @if_congr _ _ _ _ (id _) _ _ _ _ Iff.rfl rfl rfl

-- Porting note: needed unfold for mulIndicator

@[to_additive]
theorem mulIndicator_apply (s : Set α) (f : α → M) (a : α) [Decidable (a ∈ s)] :
    mulIndicator s f a = if a ∈ s then f a else 1 := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝¹ : One M
    s : Set α
    f : α → M
    a : α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (s.mulIndicator f a) (ite (Membership.mem s a) (f a) 1)
  -/
  unfold mulIndicator
  /-
    α : Type u_1
    M : Type u_3
    inst✝¹ : One M
    s : Set α
    f : α → M
    a : α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (ite (Membership.mem s a) (f a) 1) (ite (Membership.mem s a) (f a) 1)
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulIndicator_of_mem (h : a ∈ s) (f : α → M) : mulIndicator s f a = f a :=
  if_pos h


@[to_additive (attr := simp)]
theorem mulIndicator_of_not_mem (h : a ∉ s) (f : α → M) : mulIndicator s f a = 1 :=
  if_neg h


@[to_additive]
theorem mulIndicator_eq_one_or_self (s : Set α) (f : α → M) (a : α) :
    mulIndicator s f a = 1 ∨ mulIndicator s f a = f a := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : α → M
    a : α
    ⊢ Or (Eq (s.mulIndicator f a) 1) (Eq (s.mulIndicator f a) (f a))
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s : Set α
      f : α → M
      a : α
      h : Membership.mem s a
      ⊢ Or (Eq (s.mulIndicator f a) 1) (Eq (s.mulIndicator f a) (f a))
    -/
  · exact Or.inr (mulIndicator_of_mem h f)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s : Set α
      f : α → M
      a : α
      h : Not (Membership.mem s a)
      ⊢ Or (Eq (s.mulIndicator f a) 1) (Eq (s.mulIndicator f a) (f a))
    -/
  · exact Or.inl (mulIndicator_of_not_mem h f)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem mulIndicator_apply_eq_self : s.mulIndicator f a = f a ↔ a ∉ s → f a = 1 :=
  letI := Classical.dec (a ∈ s)
                            /-
                              α : Type u_1
                              M : Type u_3
                              inst✝ : One M
                              s : Set α
                              f : α → M
                              a : α
                              this : Decidable (Membership.mem s a) := Classical.dec (Membership.mem s a)
                              ⊢ Iff (Not (Membership.mem s a) → Eq 1 (f a)) (Not (Membership.mem s a) → Eq ( …
                            -/
  ite_eq_left_iff.trans (by rw [@eq_comm _ (f a)])
                            /-
                              🎉 no goals
                            -/


@[to_additive (attr := simp)]
theorem mulIndicator_eq_self : s.mulIndicator f = f ↔ mulSupport f ⊆ s := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : α → M
    ⊢ Iff (Eq (s.mulIndicator f) f) (HasSubset.Subset (Function.mulSupport f) s)
  -/
  simp only [funext_iff, subset_def, mem_mulSupport, mulIndicator_apply_eq_self, not_imp_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_eq_self_of_superset (h1 : s.mulIndicator f = f) (h2 : s ⊆ t) :
    t.mulIndicator f = f := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s t : Set α
    f : α → M
    h1 : Eq (s.mulIndicator f) f
    h2 : HasSubset.Subset s t
    ⊢ Eq (t.mulIndicator f) f
  -/
  rw [mulIndicator_eq_self] at h1 ⊢
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s t : Set α
    f : α → M
    h1 : HasSubset.Subset (Function.mulSupport f) s
    h2 : HasSubset.Subset s t
    ⊢ HasSubset.Subset (Function.mulSupport f) t
  -/
  exact Subset.trans h1 h2
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulIndicator_apply_eq_one : mulIndicator s f a = 1 ↔ a ∈ s → f a = 1 :=
  letI := Classical.dec (a ∈ s)
  ite_eq_right_iff


@[to_additive (attr := simp)]
theorem mulIndicator_eq_one : (mulIndicator s f = fun _ => 1) ↔ Disjoint (mulSupport f) s := by
  simp only [funext_iff, mulIndicator_apply_eq_one, Set.disjoint_left, mem_mulSupport,
    not_imp_not]


@[to_additive (attr := simp)]
theorem mulIndicator_eq_one' : mulIndicator s f = 1 ↔ Disjoint (mulSupport f) s :=
  mulIndicator_eq_one


@[to_additive]
theorem mulIndicator_apply_ne_one {a : α} : s.mulIndicator f a ≠ 1 ↔ a ∈ s ∩ mulSupport f := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : α → M
    a : α
    ⊢ Iff (Ne (s.mulIndicator f a) 1) (Membership.mem (Inter.inter s (Function.mul …
  -/
  simp only [Ne, mulIndicator_apply_eq_one, Classical.not_imp, mem_inter_iff, mem_mulSupport]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulSupport_mulIndicator :
    Function.mulSupport (s.mulIndicator f) = s ∩ Function.mulSupport f :=
                  /-
                    α : Type u_1
                    M : Type u_3
                    inst✝ : One M
                    s : Set α
                    f : α → M
                    x : α
                    ⊢ Iff (Membership.mem (Function.mulSupport (s.mulIndicator f)) x) (Membership. …
                  -/
  ext fun x => by simp [Function.mem_mulSupport, mulIndicator_apply_eq_one]
                  /-
                    🎉 no goals
                  -/


/-- If a multiplicative indicator function is not equal to `1` at a point, then that point is in the
set. -/
@[to_additive
      "If an additive indicator function is not equal to `0` at a point, then that point is
      in the set."]
theorem mem_of_mulIndicator_ne_one (h : mulIndicator s f a ≠ 1) : a ∈ s :=
  not_imp_comm.1 (fun hn => mulIndicator_of_not_mem hn f) h


/-- See `Set.eqOn_mulIndicator'` for the version with `sᶜ`. -/
@[to_additive
      "See `Set.eqOn_indicator'` for the version with `sᶜ`"]
theorem eqOn_mulIndicator : EqOn (mulIndicator s f) f s := fun _ hx => mulIndicator_of_mem hx f


/-- See `Set.eqOn_mulIndicator` for the version with `s`. -/
@[to_additive
      "See `Set.eqOn_indicator` for the version with `s`."]
theorem eqOn_mulIndicator' : EqOn (mulIndicator s f) 1 sᶜ :=
  fun _ hx => mulIndicator_of_not_mem hx f


@[to_additive]
theorem mulSupport_mulIndicator_subset : mulSupport (s.mulIndicator f) ⊆ s := fun _ hx =>
  hx.imp_symm fun h => mulIndicator_of_not_mem h f


@[to_additive (attr := simp)]
theorem mulIndicator_mulSupport : mulIndicator (mulSupport f) f = f :=
  mulIndicator_eq_self.2 Subset.rfl


@[to_additive (attr := simp)]
theorem mulIndicator_range_comp {ι : Sort*} (f : ι → α) (g : α → M) :
    mulIndicator (range f) g ∘ f = g ∘ f :=
  letI := Classical.decPred (· ∈ range f)
  piecewise_range_comp _ _ _


@[to_additive]
theorem mulIndicator_congr (h : EqOn f g s) : mulIndicator s f = mulIndicator s g :=
  funext fun x => by
    /-
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s : Set α
      f g : α → M
      h : Set.EqOn f g s
      x : α
      ⊢ Eq (s.mulIndicator f x) (s.mulIndicator g x)
    -/
    simp only [mulIndicator]
    /-
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s : Set α
      f g : α → M
      h : Set.EqOn f g s
      x : α
      ⊢ Eq (ite (Membership.mem s x) (f x) 1) (ite (Membership.mem s x) (g x) 1)
    -/
    split_ifs with h_1
      /-
        case pos
        α : Type u_1
        M : Type u_3
        inst✝ : One M
        s : Set α
        f g : α → M
        h : Set.EqOn f g s
        x : α
        h_1 : Membership.mem s x
        ⊢ Eq (f x) (g x)
      -/
    · exact h h_1
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s : Set α
      f g : α → M
      h : Set.EqOn f g s
      x : α
      h_1 : Not (Membership.mem s x)
      ⊢ Eq 1 1
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem mulIndicator_univ (f : α → M) : mulIndicator (univ : Set α) f = f :=
  mulIndicator_eq_self.2 <| subset_univ _


@[to_additive (attr := simp)]
theorem mulIndicator_empty (f : α → M) : mulIndicator (∅ : Set α) f = fun _ => 1 :=
  mulIndicator_eq_one.2 <| disjoint_empty _


@[to_additive]
theorem mulIndicator_empty' (f : α → M) : mulIndicator (∅ : Set α) f = 1 :=
  mulIndicator_empty f


@[to_additive (attr := simp)]
theorem mulIndicator_one (s : Set α) : (mulIndicator s fun _ => (1 : M)) = fun _ => (1 : M) :=
                              /-
                                α : Type u_1
                                M : Type u_3
                                inst✝ : One M
                                s : Set α
                                ⊢ Disjoint (Function.mulSupport fun x => 1) s
                              -/
  mulIndicator_eq_one.2 <| by simp only [mulSupport_one, empty_disjoint]
                              /-
                                🎉 no goals
                              -/


@[to_additive (attr := simp)]
theorem mulIndicator_one' {s : Set α} : s.mulIndicator (1 : α → M) = 1 :=
  mulIndicator_one M s


@[to_additive]
theorem mulIndicator_mulIndicator (s t : Set α) (f : α → M) :
    mulIndicator s (mulIndicator t f) = mulIndicator (s ∩ t) f :=
  funext fun x => by
    /-
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s t : Set α
      f : α → M
      x : α
      ⊢ Eq (s.mulIndicator (t.mulIndicator f) x) ((Inter.inter s t).mulIndicator f x)
    -/
    simp only [mulIndicator]
    /-
      α : Type u_1
      M : Type u_3
      inst✝ : One M
      s t : Set α
      f : α → M
      x : α
      ⊢ Eq (ite (Membership.mem s x) (ite (Membership.mem t x) (f x) 1) 1) (ite (Mem …
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
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> simp_all +contextual
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem mulIndicator_inter_mulSupport (s : Set α) (f : α → M) :
    mulIndicator (s ∩ mulSupport f) f = mulIndicator s f := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : α → M
    ⊢ Eq ((Inter.inter s (Function.mulSupport f)).mulIndicator f) (s.mulIndicator f)
  -/
  rw [← mulIndicator_mulIndicator, mulIndicator_mulSupport]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comp_mulIndicator (h : M → β) (f : α → M) {s : Set α} {x : α} [DecidablePred (· ∈ s)] :
    h (s.mulIndicator f x) = s.piecewise (h ∘ f) (const α (h 1)) x := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝¹ : One M
    h : M → β
    f : α → M
    s : Set α
    x : α
    inst✝ : DecidablePred fun x => Membership.mem s x
    ⊢ Eq (h (s.mulIndicator f x)) (s.piecewise (Function.comp h f) (Function.const …
  -/
  letI := Classical.decPred (· ∈ s)
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝¹ : One M
    h : M → β
    f : α → M
    s : Set α
    x : α
    inst✝ : DecidablePred fun x => Membership.mem s x
    this : DecidablePred fun x => Membership.mem s x := Classical.decPred fun x => …
    ⊢ Eq (h (s.mulIndicator f x)) (s.piecewise (Function.comp h f) (Function.const …
  -/
  convert s.apply_piecewise f (const α 1) (fun _ => h) (x := x) using 2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_comp_right {s : Set α} (f : β → α) {g : α → M} {x : β} :
    mulIndicator (f ⁻¹' s) (g ∘ f) x = mulIndicator s g (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : β → α
    g : α → M
    x : β
    ⊢ Eq ((Set.preimage f s).mulIndicator (Function.comp g f) x) (s.mulIndicator g …
  -/
  simp only [mulIndicator, Function.comp]
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : β → α
    g : α → M
    x : β
    ⊢ Eq (ite (Membership.mem (Set.preimage f s) x) (g (f x)) 1) (ite (Membership. …
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
  split_ifs with h h' h'' <;> first | rfl | contradiction
                              /-
                                🎉 no goals
                              -/


@[to_additive]
theorem mulIndicator_image {s : Set α} {f : β → M} {g : α → β} (hg : Injective g) {x : α} :
    mulIndicator (g '' s) f (g x) = mulIndicator s (f ∘ g) x := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : β → M
    g : α → β
    hg : Function.Injective g
    x : α
    ⊢ Eq ((Set.image g s).mulIndicator f (g x)) (s.mulIndicator (Function.comp f g …
  -/
  rw [← mulIndicator_comp_right, preimage_image_eq _ hg]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_comp_of_one {g : M → N} (hg : g 1 = 1) :
    mulIndicator s (g ∘ f) = g ∘ mulIndicator s f := by
  /-
    α : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹ : One M
    inst✝ : One N
    s : Set α
    f : α → M
    g : M → N
    hg : Eq (g 1) 1
    ⊢ Eq (s.mulIndicator (Function.comp g f)) (Function.comp g (s.mulIndicator f))
  -/
  funext
  /-
    case h
    α : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹ : One M
    inst✝ : One N
    s : Set α
    f : α → M
    g : M → N
    hg : Eq (g 1) 1
    x✝ : α
    ⊢ Eq (s.mulIndicator (Function.comp g f) x✝) (Function.comp g (s.mulIndicator  …
  -/
  simp only [mulIndicator]
  /-
    case h
    α : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹ : One M
    inst✝ : One N
    s : Set α
    f : α → M
    g : M → N
    hg : Eq (g 1) 1
    x✝ : α
    ⊢ Eq (ite (Membership.mem s x✝) (Function.comp g f x✝) 1) (Function.comp g (s. …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [*]
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem comp_mulIndicator_const (c : M) (f : M → N) (hf : f 1 = 1) :
    (fun x => f (s.mulIndicator (fun _ => c) x)) = s.mulIndicator fun _ => f c :=
  (mulIndicator_comp_of_one hf).symm


@[to_additive]
theorem mulIndicator_preimage (s : Set α) (f : α → M) (B : Set M) :
    mulIndicator s f ⁻¹' B = s.ite (f ⁻¹' B) (1 ⁻¹' B) :=
  letI := Classical.decPred (· ∈ s)
  piecewise_preimage s f 1 B


@[to_additive]
theorem mulIndicator_one_preimage (s : Set M) :
    t.mulIndicator 1 ⁻¹' s ∈ ({Set.univ, ∅} : Set (Set α)) := by
  classical
    rw [mulIndicator_one', preimage_one]
    split_ifs <;> simp


@[to_additive]
theorem mulIndicator_const_preimage_eq_union (U : Set α) (s : Set M) (a : M) [Decidable (a ∈ s)]
    [Decidable ((1 : M) ∈ s)] : (U.mulIndicator fun _ => a) ⁻¹' s =
      (if a ∈ s then U else ∅) ∪ if (1 : M) ∈ s then Uᶜ else ∅ := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝² : One M
    U : Set α
    s : Set M
    a : M
    inst✝¹ : Decidable (Membership.mem s a)
    inst✝ : Decidable (Membership.mem s 1)
    ⊢ Eq (Set.preimage (U.mulIndicator fun x => a) s) (Union.union (ite (Membershi …
  -/
  rw [mulIndicator_preimage, preimage_one, preimage_const]
  /-
    α : Type u_1
    M : Type u_3
    inst✝² : One M
    U : Set α
    s : Set M
    a : M
    inst✝¹ : Decidable (Membership.mem s a)
    inst✝ : Decidable (Membership.mem s 1)
    ⊢ Eq (U.ite (ite (Membership.mem s a) Set.univ EmptyCollection.emptyCollection …
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
  split_ifs <;> simp [← compl_eq_univ_diff]
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem mulIndicator_const_preimage (U : Set α) (s : Set M) (a : M) :
    (U.mulIndicator fun _ => a) ⁻¹' s ∈ ({Set.univ, U, Uᶜ, ∅} : Set (Set α)) := by
  classical
    rw [mulIndicator_const_preimage_eq_union]
    split_ifs <;> simp


theorem indicator_one_preimage [Zero M] (U : Set α) (s : Set M) :
    U.indicator 1 ⁻¹' s ∈ ({Set.univ, U, Uᶜ, ∅} : Set (Set α)) :=
  indicator_const_preimage _ _ 1


@[to_additive]
theorem mulIndicator_preimage_of_not_mem (s : Set α) (f : α → M) {t : Set M} (ht : (1 : M) ∉ t) :
    mulIndicator s f ⁻¹' t = f ⁻¹' t ∩ s := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f : α → M
    t : Set M
    ht : Not (Membership.mem t 1)
    ⊢ Eq (Set.preimage (s.mulIndicator f) t) (Inter.inter (Set.preimage f t) s)
  -/
  simp [mulIndicator_preimage, Pi.one_def, Set.preimage_const_of_not_mem ht]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_range_mulIndicator {r : M} {s : Set α} {f : α → M} :
    r ∈ range (mulIndicator s f) ↔ r = 1 ∧ s ≠ univ ∨ r ∈ f '' s := by
  simp [mulIndicator, ite_eq_iff, exists_or, eq_univ_iff_forall, and_comm, or_comm,
    @eq_comm _ r 1]


@[to_additive]
theorem mulIndicator_rel_mulIndicator {r : M → M → Prop} (h1 : r 1 1) (ha : a ∈ s → r (f a) (g a)) :
    r (mulIndicator s f a) (mulIndicator s g a) := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f g : α → M
    a : α
    r : M → M → Prop
    h1 : r 1 1
    ha : Membership.mem s a → r (f a) (g a)
    ⊢ r (s.mulIndicator f a) (s.mulIndicator g a)
  -/
  simp only [mulIndicator]
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f g : α → M
    a : α
    r : M → M → Prop
    h1 : r 1 1
    ha : Membership.mem s a → r (f a) (g a)
    ⊢ r (ite (Membership.mem s a) (f a) 1) (ite (Membership.mem s a) (g a) 1)
  -/
  split_ifs with has
  /-
    case pos
    α : Type u_1
    M : Type u_3
    inst✝ : One M
    s : Set α
    f g : α → M
    a : α
    r : M → M → Prop
    h1 : r 1 1
    ha : Membership.mem s a → r (f a) (g a)
    has : Membership.mem s a
    ⊢ r (f a) (g a)
  -/
  exacts [ha has, h1]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_union_mul_inter_apply (f : α → M) (s t : Set α) (a : α) :
    mulIndicator (s ∪ t) f a * mulIndicator (s ∩ t) f a
      = mulIndicator s f a * mulIndicator t f a := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f : α → M
    s t : Set α
    a : α
    ⊢ Eq (HMul.hMul ((Union.union s t).mulIndicator f a) ((Inter.inter s t).mulInd …
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
  by_cases hs : a ∈ s <;> by_cases ht : a ∈ t <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive]
theorem mulIndicator_union_mul_inter (f : α → M) (s t : Set α) :
    mulIndicator (s ∪ t) f * mulIndicator (s ∩ t) f = mulIndicator s f * mulIndicator t f :=
  funext <| mulIndicator_union_mul_inter_apply f s t


@[to_additive]
theorem mulIndicator_union_of_not_mem_inter (h : a ∉ s ∩ t) (f : α → M) :
    mulIndicator (s ∪ t) f a = mulIndicator s f a * mulIndicator t f a := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    s t : Set α
    a : α
    h : Not (Membership.mem (Inter.inter s t) a)
    f : α → M
    ⊢ Eq ((Union.union s t).mulIndicator f a) (HMul.hMul (s.mulIndicator f a) (t.m …
  -/
  rw [← mulIndicator_union_mul_inter_apply f s t, mulIndicator_of_not_mem h, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_union_of_disjoint (h : Disjoint s t) (f : α → M) :
    mulIndicator (s ∪ t) f = fun a => mulIndicator s f a * mulIndicator t f a :=
  funext fun _ => mulIndicator_union_of_not_mem_inter (fun ha => h.le_bot ha) _


open scoped symmDiff in
@[to_additive]
theorem mulIndicator_symmDiff (s t : Set α) (f : α → M) :
    mulIndicator (s ∆ t) f = mulIndicator (s \ t) f * mulIndicator (t \ s) f :=
  mulIndicator_union_of_disjoint (disjoint_sdiff_self_right.mono_left sdiff_le) _


@[to_additive]
theorem mulIndicator_mul (s : Set α) (f g : α → M) :
    (mulIndicator s fun a => f a * g a) = fun a => mulIndicator s f a * mulIndicator s g a := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    s : Set α
    f g : α → M
    ⊢ Eq (s.mulIndicator fun a => HMul.hMul (f a) (g a)) fun a => HMul.hMul (s.mul …
  -/
  funext
  /-
    case h
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    s : Set α
    f g : α → M
    x✝ : α
    ⊢ Eq (s.mulIndicator (fun a => HMul.hMul (f a) (g a)) x✝) (HMul.hMul (s.mulInd …
  -/
  simp only [mulIndicator]
  /-
    case h
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    s : Set α
    f g : α → M
    x✝ : α
    ⊢ Eq (ite (Membership.mem s x✝) (HMul.hMul (f x✝) (g x✝)) 1) (HMul.hMul (ite ( …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      M : Type u_3
      inst✝ : MulOneClass M
      s : Set α
      f g : α → M
      x✝ : α
      h✝ : Membership.mem s x✝
      ⊢ Eq (HMul.hMul (f x✝) (g x✝)) (HMul.hMul (f x✝) (g x✝))
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    s : Set α
    f g : α → M
    x✝ : α
    h✝ : Not (Membership.mem s x✝)
    ⊢ Eq 1 (HMul.hMul 1 1)
  -/
  rw [mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_mul' (s : Set α) (f g : α → M) :
    mulIndicator s (f * g) = mulIndicator s f * mulIndicator s g :=
  mulIndicator_mul s f g


@[to_additive (attr := simp)]
theorem mulIndicator_compl_mul_self_apply (s : Set α) (f : α → M) (a : α) :
    mulIndicator sᶜ f a * mulIndicator s f a = f a :=
                                 /-
                                   α : Type u_1
                                   M : Type u_3
                                   inst✝ : MulOneClass M
                                   s : Set α
                                   f : α → M
                                   a : α
                                   ha : Membership.mem s a
                                   ⊢ Eq (HMul.hMul ((HasCompl.compl s).mulIndicator f a) (s.mulIndicator f a)) (f …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  by_cases (fun ha : a ∈ s => by simp [ha]) fun ha => by simp [ha]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive (attr := simp)]
theorem mulIndicator_compl_mul_self (s : Set α) (f : α → M) :
    mulIndicator sᶜ f * mulIndicator s f = f :=
  funext <| mulIndicator_compl_mul_self_apply s f


@[to_additive (attr := simp)]
theorem mulIndicator_self_mul_compl_apply (s : Set α) (f : α → M) (a : α) :
    mulIndicator s f a * mulIndicator sᶜ f a = f a :=
                                 /-
                                   α : Type u_1
                                   M : Type u_3
                                   inst✝ : MulOneClass M
                                   s : Set α
                                   f : α → M
                                   a : α
                                   ha : Membership.mem s a
                                   ⊢ Eq (HMul.hMul (s.mulIndicator f a) ((HasCompl.compl s).mulIndicator f a)) (f …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  by_cases (fun ha : a ∈ s => by simp [ha]) fun ha => by simp [ha]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive (attr := simp)]
theorem mulIndicator_self_mul_compl (s : Set α) (f : α → M) :
    mulIndicator s f * mulIndicator sᶜ f = f :=
  funext <| mulIndicator_self_mul_compl_apply s f


@[to_additive]
theorem mulIndicator_mul_eq_left {f g : α → M} (h : Disjoint (mulSupport f) (mulSupport g)) :
    (mulSupport f).mulIndicator (f * g) = f := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    ⊢ Eq ((Function.mulSupport f).mulIndicator (HMul.hMul f g)) f
  -/
  refine (mulIndicator_congr fun x hx => ?_).trans mulIndicator_mulSupport
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    x : α
    hx : Membership.mem (Function.mulSupport f) x
    ⊢ Eq (HMul.hMul f g x) (f x)
  -/
  have : g x = 1 := nmem_mulSupport.1 (disjoint_left.1 h hx)
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    x : α
    hx : Membership.mem (Function.mulSupport f) x
    this : Eq (g x) 1
    ⊢ Eq (HMul.hMul f g x) (f x)
  -/
  rw [Pi.mul_apply, this, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_mul_eq_right {f g : α → M} (h : Disjoint (mulSupport f) (mulSupport g)) :
    (mulSupport g).mulIndicator (f * g) = g := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    ⊢ Eq ((Function.mulSupport g).mulIndicator (HMul.hMul f g)) g
  -/
  refine (mulIndicator_congr fun x hx => ?_).trans mulIndicator_mulSupport
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    x : α
    hx : Membership.mem (Function.mulSupport g) x
    ⊢ Eq (HMul.hMul f g x) (g x)
  -/
  have : f x = 1 := nmem_mulSupport.1 (disjoint_right.1 h hx)
  /-
    α : Type u_1
    M : Type u_3
    inst✝ : MulOneClass M
    f g : α → M
    h : Disjoint (Function.mulSupport f) (Function.mulSupport g)
    x : α
    hx : Membership.mem (Function.mulSupport g) x
    this : Eq (f x) 1
    ⊢ Eq (HMul.hMul f g x) (g x)
  -/
  rw [Pi.mul_apply, this, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulIndicator_mul_compl_eq_piecewise [DecidablePred (· ∈ s)] (f g : α → M) :
    s.mulIndicator f * sᶜ.mulIndicator g = s.piecewise f g := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝¹ : MulOneClass M
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    f g : α → M
    ⊢ Eq (HMul.hMul (s.mulIndicator f) ((HasCompl.compl s).mulIndicator g)) (s.pie …
  -/
  ext x
  /-
    case h
    α : Type u_1
    M : Type u_3
    inst✝¹ : MulOneClass M
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    f g : α → M
    x : α
    ⊢ Eq (HMul.hMul (s.mulIndicator f) ((HasCompl.compl s).mulIndicator g) x) (s.p …
  -/
  by_cases h : x ∈ s
  · rw [piecewise_eq_of_mem _ _ _ h, Pi.mul_apply, Set.mulIndicator_of_mem h,
      Set.mulIndicator_of_not_mem (Set.not_mem_compl_iff.2 h), mul_one]
  · rw [piecewise_eq_of_not_mem _ _ _ h, Pi.mul_apply, Set.mulIndicator_of_not_mem h,
      Set.mulIndicator_of_mem (Set.mem_compl h), one_mul]


/-- `Set.mulIndicator` as a `monoidHom`. -/
@[to_additive "`Set.indicator` as an `addMonoidHom`."]
noncomputable def mulIndicatorHom {α} (M) [MulOneClass M] (s : Set α) : (α → M) →* α → M where
  toFun := mulIndicator s
  map_one' := mulIndicator_one M s
  map_mul' := mulIndicator_mul s


@[to_additive]
theorem mulIndicator_inv' (s : Set α) (f : α → G) : mulIndicator s f⁻¹ = (mulIndicator s f)⁻¹ :=
  (mulIndicatorHom G s).map_inv f


@[to_additive]
theorem mulIndicator_inv (s : Set α) (f : α → G) :
    (mulIndicator s fun a => (f a)⁻¹) = fun a => (mulIndicator s f a)⁻¹ :=
  mulIndicator_inv' s f


@[to_additive]
theorem mulIndicator_div (s : Set α) (f g : α → G) :
    (mulIndicator s fun a => f a / g a) = fun a => mulIndicator s f a / mulIndicator s g a :=
  (mulIndicatorHom G s).map_div f g


@[to_additive]
theorem mulIndicator_div' (s : Set α) (f g : α → G) :
    mulIndicator s (f / g) = mulIndicator s f / mulIndicator s g :=
  mulIndicator_div s f g


@[to_additive indicator_compl']
theorem mulIndicator_compl (s : Set α) (f : α → G) :
    mulIndicator sᶜ f = f * (mulIndicator s f)⁻¹ :=
  eq_mul_inv_of_mul_eq <| s.mulIndicator_compl_mul_self f


@[to_additive indicator_compl]
theorem mulIndicator_compl' (s : Set α) (f : α → G) :
                                                   /-
                                                     α : Type u_1
                                                     G : Type u_5
                                                     inst✝ : Group G
                                                     s : Set α
                                                     f : α → G
                                                     ⊢ Eq ((HasCompl.compl s).mulIndicator f) (HDiv.hDiv f (s.mulIndicator f))
                                                   -/
    mulIndicator sᶜ f = f / mulIndicator s f := by rw [div_eq_mul_inv, mulIndicator_compl]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive indicator_diff']
theorem mulIndicator_diff (h : s ⊆ t) (f : α → G) :
    mulIndicator (t \ s) f = mulIndicator t f * (mulIndicator s f)⁻¹ :=
  eq_mul_inv_of_mul_eq <| by
    rw [Pi.mul_def, ← mulIndicator_union_of_disjoint, diff_union_self,
      union_eq_self_of_subset_right h]
    /-
      case h
      α : Type u_1
      G : Type u_5
      inst✝ : Group G
      s t : Set α
      h : HasSubset.Subset s t
      f : α → G
      ⊢ Disjoint (SDiff.sdiff t s) s
    -/
    exact disjoint_sdiff_self_left
    /-
      🎉 no goals
    -/


@[to_additive indicator_diff]
theorem mulIndicator_diff' (h : s ⊆ t) (f : α → G) :
    mulIndicator (t \ s) f = mulIndicator t f / mulIndicator s f := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : Group G
    s t : Set α
    h : HasSubset.Subset s t
    f : α → G
    ⊢ Eq ((SDiff.sdiff t s).mulIndicator f) (HDiv.hDiv (t.mulIndicator f) (s.mulIn …
  -/
  rw [mulIndicator_diff h, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


open scoped symmDiff in
@[to_additive]
theorem apply_mulIndicator_symmDiff {g : G → β} (hg : ∀ x, g x⁻¹ = g x)
    (s t : Set α) (f : α → G) (x : α) :
    g (mulIndicator (s ∆ t) f x) = g (mulIndicator s f x / mulIndicator t f x) := by
  /-
    α : Type u_1
    β : Type u_2
    G : Type u_5
    inst✝ : Group G
    g : G → β
    hg : ∀ (x : G), Eq (g (Inv.inv x)) (g x)
    s t : Set α
    f : α → G
    x : α
    ⊢ Eq (g ((symmDiff s t).mulIndicator f x)) (g (HDiv.hDiv (s.mulIndicator f x)  …
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
  by_cases hs : x ∈ s <;> by_cases ht : x ∈ t <;> simp [mem_symmDiff, *]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive]
theorem MonoidHom.map_mulIndicator {M N : Type*} [MulOneClass M] [MulOneClass N] (f : M →* N)
    (s : Set α) (g : α → M) (x : α) : f (s.mulIndicator g x) = s.mulIndicator (f ∘ g) x := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_6
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass N
    f : MonoidHom M N
    s : Set α
    g : α → M
    x : α
    ⊢ Eq (f (s.mulIndicator g x)) (s.mulIndicator (Function.comp (⇑f) g) x)
  -/
  simp [Set.mulIndicator_comp_of_one]
  /-
    🎉 no goals
  -/


/-- Given two finitely supported functions `f g : α →₀ N`, `Finsupp.neLocus f g` is the `Finset`
where `f` and `g` differ. This generalizes `(f - g).support` to situations without subtraction. -/
def neLocus (f g : Π₀ a, N a) : Finset α :=
  (f.support ∪ g.support).filter fun x ↦ f x ≠ g x


@[simp]
theorem mem_neLocus {f g : Π₀ a, N a} {a : α} : a ∈ f.neLocus g ↔ f a ≠ g a := by
  simpa only [neLocus, Finset.mem_filter, Finset.mem_union, mem_support_iff,
    and_iff_right_iff_imp] using Ne.ne_or_ne _


theorem not_mem_neLocus {f g : Π₀ a, N a} {a : α} : a ∉ f.neLocus g ↔ f a = g a :=
  mem_neLocus.not.trans not_ne_iff


@[simp]
theorem coe_neLocus : ↑(f.neLocus g) = { x | f x ≠ g x } :=
  Set.ext fun _x ↦ mem_neLocus


@[simp]
theorem neLocus_eq_empty {f g : Π₀ a, N a} : f.neLocus g = ∅ ↔ f = g :=
  ⟨fun h ↦
    ext fun a ↦ not_not.mp (mem_neLocus.not.mp (Finset.eq_empty_iff_forall_not_mem.mp h a)),
                   /-
                     α : Type u_1
                     N : α → Type u_2
                     inst✝² : DecidableEq α
                     inst✝¹ : (a : α) → DecidableEq (N a)
                     inst✝ : (a : α) → Zero (N a)
                     f g : DFinsupp fun a => N a
                     h : Eq f g
                     ⊢ Eq (f.neLocus f) EmptyCollection.emptyCollection
                   -/
    fun h ↦ h ▸ by simp only [neLocus, Ne, eq_self_iff_true, not_true, Finset.filter_False]⟩
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem nonempty_neLocus_iff {f g : Π₀ a, N a} : (f.neLocus g).Nonempty ↔ f ≠ g :=
  Finset.nonempty_iff_ne_empty.trans neLocus_eq_empty.not


theorem neLocus_comm : f.neLocus g = g.neLocus f := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → Zero (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq (f.neLocus g) (g.neLocus f)
  -/
  simp_rw [neLocus, Finset.union_comm, ne_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_zero_right : f.neLocus 0 = f.support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → Zero (N a)
    f : DFinsupp fun a => N a
    ⊢ Eq (f.neLocus 0) f.support
  -/
  ext
  /-
    case h
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → Zero (N a)
    f : DFinsupp fun a => N a
    a✝ : α
    ⊢ Iff (Membership.mem (f.neLocus 0) a✝) (Membership.mem f.support a✝)
  -/
  rw [mem_neLocus, mem_support_iff, coe_zero, Pi.zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_zero_left : (0 : Π₀ a, N a).neLocus f = f.support :=
  (neLocus_comm _ _).trans (neLocus_zero_right _)


theorem subset_mapRange_neLocus [∀ a, DecidableEq (N a)] [∀ a, DecidableEq (M a)] (f g : Π₀ a, N a)
    {F : ∀ a, N a → M a} (F0 : ∀ a, F a 0 = 0) :
    (f.mapRange F F0).neLocus (g.mapRange F F0) ⊆ f.neLocus g := fun a ↦ by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝⁴ : DecidableEq α
    M : α → Type u_3
    inst✝³ : (a : α) → Zero (N a)
    inst✝² : (a : α) → Zero (M a)
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → DecidableEq (M a)
    f g : DFinsupp fun a => N a
    F : (a : α) → N a → M a
    F0 : ∀ (a : α), Eq (F a 0) 0
    a : α
    ⊢ Membership.mem ((DFinsupp.mapRange F F0 f).neLocus (DFinsupp.mapRange F F0 g …
  -/
  simpa only [mem_neLocus, mapRange_apply, not_imp_not] using congr_arg (F a)
  /-
    🎉 no goals
  -/


theorem zipWith_neLocus_eq_left [∀ a, DecidableEq (N a)] [∀ a, DecidableEq (P a)]
    {F : ∀ a, M a → N a → P a} (F0 : ∀ a, F a 0 0 = 0) (f : Π₀ a, M a) (g₁ g₂ : Π₀ a, N a)
    (hF : ∀ a f, Function.Injective fun g ↦ F a f g) :
    (zipWith F F0 f g₁).neLocus (zipWith F F0 f g₂) = g₁.neLocus g₂ := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝⁵ : DecidableEq α
    M : α → Type u_3
    P : α → Type u_4
    inst✝⁴ : (a : α) → Zero (N a)
    inst✝³ : (a : α) → Zero (M a)
    inst✝² : (a : α) → Zero (P a)
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → DecidableEq (P a)
    F : (a : α) → M a → N a → P a
    F0 : ∀ (a : α), Eq (F a 0 0) 0
    f : DFinsupp fun a => M a
    g₁ g₂ : DFinsupp fun a => N a
    hF : ∀ (a : α) (f : M a), Function.Injective fun g => F a f g
    ⊢ Eq ((DFinsupp.zipWith F F0 f g₁).neLocus (DFinsupp.zipWith F F0 f g₂)) (g₁.n …
  -/
  ext a
  /-
    case h
    α : Type u_1
    N : α → Type u_2
    inst✝⁵ : DecidableEq α
    M : α → Type u_3
    P : α → Type u_4
    inst✝⁴ : (a : α) → Zero (N a)
    inst✝³ : (a : α) → Zero (M a)
    inst✝² : (a : α) → Zero (P a)
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → DecidableEq (P a)
    F : (a : α) → M a → N a → P a
    F0 : ∀ (a : α), Eq (F a 0 0) 0
    f : DFinsupp fun a => M a
    g₁ g₂ : DFinsupp fun a => N a
    hF : ∀ (a : α) (f : M a), Function.Injective fun g => F a f g
    a : α
    ⊢ Iff (Membership.mem ((DFinsupp.zipWith F F0 f g₁).neLocus (DFinsupp.zipWith  …
  -/
  simpa only [mem_neLocus] using (hF a _).ne_iff
  /-
    🎉 no goals
  -/


theorem zipWith_neLocus_eq_right [∀ a, DecidableEq (M a)] [∀ a, DecidableEq (P a)]
    {F : ∀ a, M a → N a → P a} (F0 : ∀ a, F a 0 0 = 0) (f₁ f₂ : Π₀ a, M a) (g : Π₀ a, N a)
    (hF : ∀ a g, Function.Injective fun f ↦ F a f g) :
    (zipWith F F0 f₁ g).neLocus (zipWith F F0 f₂ g) = f₁.neLocus f₂ := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝⁵ : DecidableEq α
    M : α → Type u_3
    P : α → Type u_4
    inst✝⁴ : (a : α) → Zero (N a)
    inst✝³ : (a : α) → Zero (M a)
    inst✝² : (a : α) → Zero (P a)
    inst✝¹ : (a : α) → DecidableEq (M a)
    inst✝ : (a : α) → DecidableEq (P a)
    F : (a : α) → M a → N a → P a
    F0 : ∀ (a : α), Eq (F a 0 0) 0
    f₁ f₂ : DFinsupp fun a => M a
    g : DFinsupp fun a => N a
    hF : ∀ (a : α) (g : N a), Function.Injective fun f => F a f g
    ⊢ Eq ((DFinsupp.zipWith F F0 f₁ g).neLocus (DFinsupp.zipWith F F0 f₂ g)) (f₁.n …
  -/
  ext a
  /-
    case h
    α : Type u_1
    N : α → Type u_2
    inst✝⁵ : DecidableEq α
    M : α → Type u_3
    P : α → Type u_4
    inst✝⁴ : (a : α) → Zero (N a)
    inst✝³ : (a : α) → Zero (M a)
    inst✝² : (a : α) → Zero (P a)
    inst✝¹ : (a : α) → DecidableEq (M a)
    inst✝ : (a : α) → DecidableEq (P a)
    F : (a : α) → M a → N a → P a
    F0 : ∀ (a : α), Eq (F a 0 0) 0
    f₁ f₂ : DFinsupp fun a => M a
    g : DFinsupp fun a => N a
    hF : ∀ (a : α) (g : N a), Function.Injective fun f => F a f g
    a : α
    ⊢ Iff (Membership.mem ((DFinsupp.zipWith F F0 f₁ g).neLocus (DFinsupp.zipWith  …
  -/
  simpa only [mem_neLocus] using (hF a _).ne_iff
  /-
    🎉 no goals
  -/


theorem mapRange_neLocus_eq [∀ a, DecidableEq (N a)] [∀ a, DecidableEq (M a)] (f g : Π₀ a, N a)
    {F : ∀ a, N a → M a} (F0 : ∀ a, F a 0 = 0) (hF : ∀ a, Function.Injective (F a)) :
    (f.mapRange F F0).neLocus (g.mapRange F F0) = f.neLocus g := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝⁴ : DecidableEq α
    M : α → Type u_3
    inst✝³ : (a : α) → Zero (N a)
    inst✝² : (a : α) → Zero (M a)
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → DecidableEq (M a)
    f g : DFinsupp fun a => N a
    F : (a : α) → N a → M a
    F0 : ∀ (a : α), Eq (F a 0) 0
    hF : ∀ (a : α), Function.Injective (F a)
    ⊢ Eq ((DFinsupp.mapRange F F0 f).neLocus (DFinsupp.mapRange F F0 g)) (f.neLocu …
  -/
  ext a
  /-
    case h
    α : Type u_1
    N : α → Type u_2
    inst✝⁴ : DecidableEq α
    M : α → Type u_3
    inst✝³ : (a : α) → Zero (N a)
    inst✝² : (a : α) → Zero (M a)
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → DecidableEq (M a)
    f g : DFinsupp fun a => N a
    F : (a : α) → N a → M a
    F0 : ∀ (a : α), Eq (F a 0) 0
    hF : ∀ (a : α), Function.Injective (F a)
    a : α
    ⊢ Iff (Membership.mem ((DFinsupp.mapRange F F0 f).neLocus (DFinsupp.mapRange F …
  -/
  simpa only [mem_neLocus] using (hF a).ne_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_add_left [∀ a, AddLeftCancelMonoid (N a)] (f g h : Π₀ a, N a) :
    (f + g).neLocus (f + h) = g.neLocus h :=
  zipWith_neLocus_eq_left _ _ _ _ fun _a ↦ add_right_injective


@[simp]
theorem neLocus_add_right [∀ a, AddRightCancelMonoid (N a)] (f g h : Π₀ a, N a) :
    (f + h).neLocus (g + h) = f.neLocus g :=
  zipWith_neLocus_eq_right _ _ _ _ fun _a ↦ add_left_injective


@[simp]
theorem neLocus_neg_neg : neLocus (-f) (-g) = f.neLocus g :=
  mapRange_neLocus_eq _ _ (fun _a ↦ neg_zero) fun _a ↦ neg_injective


                                                            /-
                                                              α : Type u_1
                                                              N : α → Type u_2
                                                              inst✝² : DecidableEq α
                                                              inst✝¹ : (a : α) → DecidableEq (N a)
                                                              inst✝ : (a : α) → AddGroup (N a)
                                                              f g : DFinsupp fun a => N a
                                                              ⊢ Eq ((Neg.neg f).neLocus g) (f.neLocus (Neg.neg g))
                                                            -/
theorem neLocus_neg : neLocus (-f) g = f.neLocus (-g) := by rw [← neLocus_neg_neg, neg_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem neLocus_eq_support_sub : f.neLocus g = (f - g).support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq (f.neLocus g) (HSub.hSub f g).support
  -/
  rw [← @neLocus_add_right α N _ _ _ _ _ (-g), add_neg_cancel, neLocus_zero_right, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_sub_left : neLocus (f - g₁) (f - g₂) = neLocus g₁ g₂ := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g₁ g₂ : DFinsupp fun a => N a
    ⊢ Eq ((HSub.hSub f g₁).neLocus (HSub.hSub f g₂)) (g₁.neLocus g₂)
  -/
  simp only [sub_eq_add_neg, @neLocus_add_left α N _ _ _, neLocus_neg_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_sub_right : neLocus (f₁ - g) (f₂ - g) = neLocus f₁ f₂ := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f₁ f₂ g : DFinsupp fun a => N a
    ⊢ Eq ((HSub.hSub f₁ g).neLocus (HSub.hSub f₂ g)) (f₁.neLocus f₂)
  -/
  simpa only [sub_eq_add_neg] using @neLocus_add_right α N _ _ _ _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_add_right : neLocus f (f + g) = g.support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq (f.neLocus (HAdd.hAdd f g)) g.support
  -/
  rw [← neLocus_zero_left, ← @neLocus_add_left α N _ _ _ f 0 g, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_add_left : neLocus (f + g) f = g.support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq ((HAdd.hAdd f g).neLocus f) g.support
  -/
  rw [neLocus_comm, neLocus_self_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_sub_right : neLocus f (f - g) = g.support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq (f.neLocus (HSub.hSub f g)) g.support
  -/
  rw [sub_eq_add_neg, neLocus_self_add_right, support_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_sub_left : neLocus (f - g) f = g.support := by
  /-
    α : Type u_1
    N : α → Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : (a : α) → DecidableEq (N a)
    inst✝ : (a : α) → AddGroup (N a)
    f g : DFinsupp fun a => N a
    ⊢ Eq ((HSub.hSub f g).neLocus f) g.support
  -/
  rw [neLocus_comm, neLocus_self_sub_right]
  /-
    🎉 no goals
  -/



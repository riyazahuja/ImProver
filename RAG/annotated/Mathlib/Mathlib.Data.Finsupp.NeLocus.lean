/-- Given two finitely supported functions `f g : α →₀ N`, `Finsupp.neLocus f g` is the `Finset`
where `f` and `g` differ. This generalizes `(f - g).support` to situations without subtraction. -/
def neLocus (f g : α →₀ N) : Finset α :=
  (f.support ∪ g.support).filter fun x => f x ≠ g x


@[simp]
theorem mem_neLocus {f g : α →₀ N} {a : α} : a ∈ f.neLocus g ↔ f a ≠ g a := by
  simpa only [neLocus, Finset.mem_filter, Finset.mem_union, mem_support_iff,
    and_iff_right_iff_imp] using Ne.ne_or_ne _


theorem not_mem_neLocus {f g : α →₀ N} {a : α} : a ∉ f.neLocus g ↔ f a = g a :=
  mem_neLocus.not.trans not_ne_iff


@[simp]
theorem coe_neLocus : ↑(f.neLocus g) = { x | f x ≠ g x } := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : Zero N
    f g : Finsupp α N
    ⊢ Eq (↑(f.neLocus g)) (setOf fun x => Ne (f x) (g x))
  -/
  ext
  /-
    case h
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : Zero N
    f g : Finsupp α N
    x✝ : α
    ⊢ Iff (Membership.mem (↑(f.neLocus g)) x✝) (Membership.mem (setOf fun x => Ne  …
  -/
  exact mem_neLocus
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_eq_empty {f g : α →₀ N} : f.neLocus g = ∅ ↔ f = g :=
  ⟨fun h =>
    ext fun a => not_not.mp (mem_neLocus.not.mp (Finset.eq_empty_iff_forall_not_mem.mp h a)),
                    /-
                      α : Type u_1
                      N : Type u_3
                      inst✝² : DecidableEq α
                      inst✝¹ : DecidableEq N
                      inst✝ : Zero N
                      f g : Finsupp α N
                      h : Eq f g
                      ⊢ Eq (f.neLocus f) EmptyCollection.emptyCollection
                    -/
    fun h => h ▸ by simp only [neLocus, Ne, eq_self_iff_true, not_true, Finset.filter_False]⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem nonempty_neLocus_iff {f g : α →₀ N} : (f.neLocus g).Nonempty ↔ f ≠ g :=
  Finset.nonempty_iff_ne_empty.trans neLocus_eq_empty.not


theorem neLocus_comm : f.neLocus g = g.neLocus f := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : Zero N
    f g : Finsupp α N
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
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : Zero N
    f : Finsupp α N
    ⊢ Eq (f.neLocus 0) f.support
  -/
  ext
  /-
    case h
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : Zero N
    f : Finsupp α N
    a✝ : α
    ⊢ Iff (Membership.mem (f.neLocus 0) a✝) (Membership.mem f.support a✝)
  -/
  rw [mem_neLocus, mem_support_iff, coe_zero, Pi.zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_zero_left : (0 : α →₀ N).neLocus f = f.support :=
  (neLocus_comm _ _).trans (neLocus_zero_right _)


theorem subset_mapRange_neLocus [DecidableEq N] [Zero N] [DecidableEq M] [Zero M] (f g : α →₀ N)
    {F : N → M} (F0 : F 0 = 0) : (f.mapRange F F0).neLocus (g.mapRange F F0) ⊆ f.neLocus g :=
              /-
                α : Type u_1
                M : Type u_2
                N : Type u_3
                inst✝⁴ : DecidableEq α
                inst✝³ : DecidableEq N
                inst✝² : Zero N
                inst✝¹ : DecidableEq M
                inst✝ : Zero M
                f g : Finsupp α N
                F : N → M
                F0 : Eq (F 0) 0
                x : α
                ⊢ Membership.mem ((Finsupp.mapRange F F0 f).neLocus (Finsupp.mapRange F F0 g)) …
              -/
  fun x => by simpa only [mem_neLocus, mapRange_apply, not_imp_not] using congr_arg F
              /-
                🎉 no goals
              -/


theorem zipWith_neLocus_eq_left [DecidableEq N] [Zero M] [DecidableEq P] [Zero P] [Zero N]
    {F : M → N → P} (F0 : F 0 0 = 0) (f : α →₀ M) (g₁ g₂ : α →₀ N)
    (hF : ∀ f, Function.Injective fun g => F f g) :
    (zipWith F F0 f g₁).neLocus (zipWith F F0 f g₂) = g₁.neLocus g₂ := by
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq N
    inst✝³ : Zero M
    inst✝² : DecidableEq P
    inst✝¹ : Zero P
    inst✝ : Zero N
    F : M → N → P
    F0 : Eq (F 0 0) 0
    f : Finsupp α M
    g₁ g₂ : Finsupp α N
    hF : ∀ (f : M), Function.Injective fun g => F f g
    ⊢ Eq ((Finsupp.zipWith F F0 f g₁).neLocus (Finsupp.zipWith F F0 f g₂)) (g₁.neL …
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq N
    inst✝³ : Zero M
    inst✝² : DecidableEq P
    inst✝¹ : Zero P
    inst✝ : Zero N
    F : M → N → P
    F0 : Eq (F 0 0) 0
    f : Finsupp α M
    g₁ g₂ : Finsupp α N
    hF : ∀ (f : M), Function.Injective fun g => F f g
    a✝ : α
    ⊢ Iff (Membership.mem ((Finsupp.zipWith F F0 f g₁).neLocus (Finsupp.zipWith F  …
  -/
  simpa only [mem_neLocus] using (hF _).ne_iff
  /-
    🎉 no goals
  -/


theorem zipWith_neLocus_eq_right [DecidableEq M] [Zero M] [DecidableEq P] [Zero P] [Zero N]
    {F : M → N → P} (F0 : F 0 0 = 0) (f₁ f₂ : α →₀ M) (g : α →₀ N)
    (hF : ∀ g, Function.Injective fun f => F f g) :
    (zipWith F F0 f₁ g).neLocus (zipWith F F0 f₂ g) = f₁.neLocus f₂ := by
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq M
    inst✝³ : Zero M
    inst✝² : DecidableEq P
    inst✝¹ : Zero P
    inst✝ : Zero N
    F : M → N → P
    F0 : Eq (F 0 0) 0
    f₁ f₂ : Finsupp α M
    g : Finsupp α N
    hF : ∀ (g : N), Function.Injective fun f => F f g
    ⊢ Eq ((Finsupp.zipWith F F0 f₁ g).neLocus (Finsupp.zipWith F F0 f₂ g)) (f₁.neL …
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq M
    inst✝³ : Zero M
    inst✝² : DecidableEq P
    inst✝¹ : Zero P
    inst✝ : Zero N
    F : M → N → P
    F0 : Eq (F 0 0) 0
    f₁ f₂ : Finsupp α M
    g : Finsupp α N
    hF : ∀ (g : N), Function.Injective fun f => F f g
    a✝ : α
    ⊢ Iff (Membership.mem ((Finsupp.zipWith F F0 f₁ g).neLocus (Finsupp.zipWith F  …
  -/
  simpa only [mem_neLocus] using (hF _).ne_iff
  /-
    🎉 no goals
  -/


theorem mapRange_neLocus_eq [DecidableEq N] [DecidableEq M] [Zero M] [Zero N] (f g : α →₀ N)
    {F : N → M} (F0 : F 0 = 0) (hF : Function.Injective F) :
    (f.mapRange F F0).neLocus (g.mapRange F F0) = f.neLocus g := by
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : DecidableEq α
    inst✝³ : DecidableEq N
    inst✝² : DecidableEq M
    inst✝¹ : Zero M
    inst✝ : Zero N
    f g : Finsupp α N
    F : N → M
    F0 : Eq (F 0) 0
    hF : Function.Injective F
    ⊢ Eq ((Finsupp.mapRange F F0 f).neLocus (Finsupp.mapRange F F0 g)) (f.neLocus g)
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : DecidableEq α
    inst✝³ : DecidableEq N
    inst✝² : DecidableEq M
    inst✝¹ : Zero M
    inst✝ : Zero N
    f g : Finsupp α N
    F : N → M
    F0 : Eq (F 0) 0
    hF : Function.Injective F
    a✝ : α
    ⊢ Iff (Membership.mem ((Finsupp.mapRange F F0 f).neLocus (Finsupp.mapRange F F …
  -/
  simpa only [mem_neLocus] using hF.ne_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_add_left [AddLeftCancelMonoid N] (f g h : α →₀ N) :
    (f + g).neLocus (f + h) = g.neLocus h :=
  zipWith_neLocus_eq_left _ _ _ _ add_right_injective


@[simp]
theorem neLocus_add_right [AddRightCancelMonoid N] (f g h : α →₀ N) :
    (f + h).neLocus (g + h) = f.neLocus g :=
  zipWith_neLocus_eq_right _ _ _ _ add_left_injective


@[simp]
theorem neLocus_neg_neg : neLocus (-f) (-g) = f.neLocus g :=
  mapRange_neLocus_eq _ _ neg_zero neg_injective


                                                            /-
                                                              α : Type u_1
                                                              N : Type u_3
                                                              inst✝² : DecidableEq α
                                                              inst✝¹ : DecidableEq N
                                                              inst✝ : AddGroup N
                                                              f g : Finsupp α N
                                                              ⊢ Eq ((Neg.neg f).neLocus g) (f.neLocus (Neg.neg g))
                                                            -/
theorem neLocus_neg : neLocus (-f) g = f.neLocus (-g) := by rw [← neLocus_neg_neg, neg_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem neLocus_eq_support_sub : f.neLocus g = (f - g).support := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g : Finsupp α N
    ⊢ Eq (f.neLocus g) (HSub.hSub f g).support
  -/
  rw [← neLocus_add_right _ _ (-g), add_neg_cancel, neLocus_zero_right, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_sub_left : neLocus (f - g₁) (f - g₂) = neLocus g₁ g₂ := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g₁ g₂ : Finsupp α N
    ⊢ Eq ((HSub.hSub f g₁).neLocus (HSub.hSub f g₂)) (g₁.neLocus g₂)
  -/
  simp only [sub_eq_add_neg, neLocus_add_left, neLocus_neg_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_sub_right : neLocus (f₁ - g) (f₂ - g) = neLocus f₁ f₂ := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f₁ f₂ g : Finsupp α N
    ⊢ Eq ((HSub.hSub f₁ g).neLocus (HSub.hSub f₂ g)) (f₁.neLocus f₂)
  -/
  simpa only [sub_eq_add_neg] using neLocus_add_right _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_add_right : neLocus f (f + g) = g.support := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g : Finsupp α N
    ⊢ Eq (f.neLocus (HAdd.hAdd f g)) g.support
  -/
  rw [← neLocus_zero_left, ← neLocus_add_left f 0 g, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem neLocus_self_add_left : neLocus (f + g) f = g.support := by
  /-
    α : Type u_1
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g : Finsupp α N
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
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g : Finsupp α N
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
    N : Type u_3
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq N
    inst✝ : AddGroup N
    f g : Finsupp α N
    ⊢ Eq ((HSub.hSub f g).neLocus f) g.support
  -/
  rw [neLocus_comm, neLocus_self_sub_right]
  /-
    🎉 no goals
  -/



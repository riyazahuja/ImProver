/-- Least common multiple of a finite set -/
def lcm (s : Finset β) (f : β → α) : α :=
  s.fold GCDMonoid.lcm 1 f


theorem lcm_def : s.lcm f = (s.1.map f).lcm :=
  rfl


@[simp]
theorem lcm_empty : (∅ : Finset β).lcm f = 1 :=
  fold_empty


@[simp]
theorem lcm_dvd_iff {a : α} : s.lcm f ∣ a ↔ ∀ b ∈ s, f b ∣ a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (Dvd.dvd (s.lcm f) a) (∀ (b : β), Membership.mem s b → Dvd.dvd (f b) a)
  -/
  apply Iff.trans Multiset.lcm_dvd
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α), Membership.mem (Multiset.map f s.val) b → Dvd.dvd b a) (∀ (b …
  -/
  simp only [Multiset.mem_map, and_imp, exists_imp]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α) (x : β), Membership.mem s.val x → Eq (f x) b → Dvd.dvd b a) ( …
  -/
  exact ⟨fun k b hb ↦ k _ _ hb rfl, fun k a' b hb h ↦ h ▸ k _ hb⟩
  /-
    🎉 no goals
  -/


theorem lcm_dvd {a : α} : (∀ b ∈ s, f b ∣ a) → s.lcm f ∣ a :=
  lcm_dvd_iff.2


theorem dvd_lcm {b : β} (hb : b ∈ s) : f b ∣ s.lcm f :=
  lcm_dvd_iff.1 dvd_rfl _ hb


@[simp]
theorem lcm_insert [DecidableEq β] {b : β} :
    (insert b s : Finset β).lcm f = GCDMonoid.lcm (f b) (s.lcm f) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    inst✝ : DecidableEq β
    b : β
    ⊢ Eq ((Insert.insert b s).lcm f) (GCDMonoid.lcm (f b) (s.lcm f))
  -/
  by_cases h : b ∈ s
  · rw [insert_eq_of_mem h,
      (lcm_eq_right_iff (f b) (s.lcm f) (Multiset.normalize_lcm (s.1.map f))).2 (dvd_lcm h)]
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    inst✝ : DecidableEq β
    b : β
    h : Not (Membership.mem s b)
    ⊢ Eq ((Insert.insert b s).lcm f) (GCDMonoid.lcm (f b) (s.lcm f))
  -/
  apply fold_insert h
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_singleton {b : β} : ({b} : Finset β).lcm f = normalize (f b) :=
  Multiset.lcm_singleton

-- Porting note: Priority changed for `simpNF`

@[simp 1100]
                                                            /-
                                                              α : Type u_2
                                                              β : Type u_3
                                                              inst✝¹ : CancelCommMonoidWithZero α
                                                              inst✝ : NormalizedGCDMonoid α
                                                              s : Finset β
                                                              f : β → α
                                                              ⊢ Eq (normalize (s.lcm f)) (s.lcm f)
                                                            -/
theorem normalize_lcm : normalize (s.lcm f) = s.lcm f := by simp [lcm_def]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem lcm_union [DecidableEq β] : (s₁ ∪ s₂).lcm f = GCDMonoid.lcm (s₁.lcm f) (s₂.lcm f) :=
                             /-
                               α : Type u_2
                               β : Type u_3
                               inst✝² : CancelCommMonoidWithZero α
                               inst✝¹ : NormalizedGCDMonoid α
                               s₁ s₂ : Finset β
                               f : β → α
                               inst✝ : DecidableEq β
                               ⊢ Eq ((Union.union EmptyCollection.emptyCollection s₂).lcm f) (GCDMonoid.lcm ( …
                             -/
  Finset.induction_on s₁ (by rw [empty_union, lcm_empty, lcm_one_left, normalize_lcm])
                             /-
                               🎉 no goals
                             -/
                      /-
                        α : Type u_2
                        β : Type u_3
                        inst✝² : CancelCommMonoidWithZero α
                        inst✝¹ : NormalizedGCDMonoid α
                        s₁ s₂ : Finset β
                        f : β → α
                        inst✝ : DecidableEq β
                        a : β
                        s : Finset β
                        x✝ : Not (Membership.mem s a)
                        ih : Eq ((Union.union s s₂).lcm f) (GCDMonoid.lcm (s.lcm f) (s₂.lcm f))
                        ⊢ Eq ((Union.union (Insert.insert a s) s₂).lcm f) (GCDMonoid.lcm ((Insert.inse …
                      -/
    fun a s _ ih ↦ by rw [insert_union, lcm_insert, lcm_insert, ih, lcm_assoc]
                      /-
                        🎉 no goals
                      -/


theorem lcm_congr {f g : β → α} (hs : s₁ = s₂) (hfg : ∀ a ∈ s₂, f a = g a) :
    s₁.lcm f = s₂.lcm g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s₁ s₂ : Finset β
    f g : β → α
    hs : Eq s₁ s₂
    hfg : ∀ (a : β), Membership.mem s₂ a → Eq (f a) (g a)
    ⊢ Eq (s₁.lcm f) (s₂.lcm g)
  -/
  subst hs
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s₁ : Finset β
    f g : β → α
    hfg : ∀ (a : β), Membership.mem s₁ a → Eq (f a) (g a)
    ⊢ Eq (s₁.lcm f) (s₁.lcm g)
  -/
  exact Finset.fold_congr hfg
  /-
    🎉 no goals
  -/


theorem lcm_mono_fun {g : β → α} (h : ∀ b ∈ s, f b ∣ g b) : s.lcm f ∣ s.lcm g :=
  lcm_dvd fun b hb ↦ (h b hb).trans (dvd_lcm hb)


theorem lcm_mono (h : s₁ ⊆ s₂) : s₁.lcm f ∣ s₂.lcm f :=
  lcm_dvd fun _ hb ↦ dvd_lcm (h hb)


theorem lcm_image [DecidableEq β] {g : γ → β} (s : Finset γ) :
    (s.image g).lcm f = s.lcm (f ∘ g) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    f : β → α
    inst✝ : DecidableEq β
    g : γ → β
    s : Finset γ
    ⊢ Eq ((Finset.image g s).lcm f) (s.lcm (Function.comp f g))
  -/
  classical induction s using Finset.induction <;> simp [*]
  /-
    🎉 no goals
  -/


theorem lcm_eq_lcm_image [DecidableEq α] : s.lcm f = (s.image f).lcm id :=
  Eq.symm <| lcm_image _


theorem lcm_eq_zero_iff [Nontrivial α] : s.lcm f = 0 ↔ 0 ∈ f '' s := by
  simp only [Multiset.mem_map, lcm_def, Multiset.lcm_eq_zero_iff, Set.mem_image, mem_coe, ←
    Finset.mem_def]


/-- Greatest common divisor of a finite set -/
def gcd (s : Finset β) (f : β → α) : α :=
  s.fold GCDMonoid.gcd 0 f


theorem gcd_def : s.gcd f = (s.1.map f).gcd :=
  rfl


@[simp]
theorem gcd_empty : (∅ : Finset β).gcd f = 0 :=
  fold_empty


theorem dvd_gcd_iff {a : α} : a ∣ s.gcd f ↔ ∀ b ∈ s, a ∣ f b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (Dvd.dvd a (s.gcd f)) (∀ (b : β), Membership.mem s b → Dvd.dvd a (f b))
  -/
  apply Iff.trans Multiset.dvd_gcd
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α), Membership.mem (Multiset.map f s.val) b → Dvd.dvd a b) (∀ (b …
  -/
  simp only [Multiset.mem_map, and_imp, exists_imp]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α) (x : β), Membership.mem s.val x → Eq (f x) b → Dvd.dvd a b) ( …
  -/
  exact ⟨fun k b hb ↦ k _ _ hb rfl, fun k a' b hb h ↦ h ▸ k _ hb⟩
  /-
    🎉 no goals
  -/


theorem gcd_dvd {b : β} (hb : b ∈ s) : s.gcd f ∣ f b :=
  dvd_gcd_iff.1 dvd_rfl _ hb


theorem dvd_gcd {a : α} : (∀ b ∈ s, a ∣ f b) → a ∣ s.gcd f :=
  dvd_gcd_iff.2


@[simp]
theorem gcd_insert [DecidableEq β] {b : β} :
    (insert b s : Finset β).gcd f = GCDMonoid.gcd (f b) (s.gcd f) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    inst✝ : DecidableEq β
    b : β
    ⊢ Eq ((Insert.insert b s).gcd f) (GCDMonoid.gcd (f b) (s.gcd f))
  -/
  by_cases h : b ∈ s
  · rw [insert_eq_of_mem h,
      (gcd_eq_right_iff (f b) (s.gcd f) (Multiset.normalize_gcd (s.1.map f))).2 (gcd_dvd h)]
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    inst✝ : DecidableEq β
    b : β
    h : Not (Membership.mem s b)
    ⊢ Eq ((Insert.insert b s).gcd f) (GCDMonoid.gcd (f b) (s.gcd f))
  -/
  apply fold_insert h
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_singleton {b : β} : ({b} : Finset β).gcd f = normalize (f b) :=
  Multiset.gcd_singleton

-- Porting note: Priority changed for `simpNF`

@[simp 1100]
                                                            /-
                                                              α : Type u_2
                                                              β : Type u_3
                                                              inst✝¹ : CancelCommMonoidWithZero α
                                                              inst✝ : NormalizedGCDMonoid α
                                                              s : Finset β
                                                              f : β → α
                                                              ⊢ Eq (normalize (s.gcd f)) (s.gcd f)
                                                            -/
theorem normalize_gcd : normalize (s.gcd f) = s.gcd f := by simp [gcd_def]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem gcd_union [DecidableEq β] : (s₁ ∪ s₂).gcd f = GCDMonoid.gcd (s₁.gcd f) (s₂.gcd f) :=
                             /-
                               α : Type u_2
                               β : Type u_3
                               inst✝² : CancelCommMonoidWithZero α
                               inst✝¹ : NormalizedGCDMonoid α
                               s₁ s₂ : Finset β
                               f : β → α
                               inst✝ : DecidableEq β
                               ⊢ Eq ((Union.union EmptyCollection.emptyCollection s₂).gcd f) (GCDMonoid.gcd ( …
                             -/
  Finset.induction_on s₁ (by rw [empty_union, gcd_empty, gcd_zero_left, normalize_gcd])
                             /-
                               🎉 no goals
                             -/
                      /-
                        α : Type u_2
                        β : Type u_3
                        inst✝² : CancelCommMonoidWithZero α
                        inst✝¹ : NormalizedGCDMonoid α
                        s₁ s₂ : Finset β
                        f : β → α
                        inst✝ : DecidableEq β
                        a : β
                        s : Finset β
                        x✝ : Not (Membership.mem s a)
                        ih : Eq ((Union.union s s₂).gcd f) (GCDMonoid.gcd (s.gcd f) (s₂.gcd f))
                        ⊢ Eq ((Union.union (Insert.insert a s) s₂).gcd f) (GCDMonoid.gcd ((Insert.inse …
                      -/
    fun a s _ ih ↦ by rw [insert_union, gcd_insert, gcd_insert, ih, gcd_assoc]
                      /-
                        🎉 no goals
                      -/


theorem gcd_congr {f g : β → α} (hs : s₁ = s₂) (hfg : ∀ a ∈ s₂, f a = g a) :
    s₁.gcd f = s₂.gcd g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s₁ s₂ : Finset β
    f g : β → α
    hs : Eq s₁ s₂
    hfg : ∀ (a : β), Membership.mem s₂ a → Eq (f a) (g a)
    ⊢ Eq (s₁.gcd f) (s₂.gcd g)
  -/
  subst hs
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s₁ : Finset β
    f g : β → α
    hfg : ∀ (a : β), Membership.mem s₁ a → Eq (f a) (g a)
    ⊢ Eq (s₁.gcd f) (s₁.gcd g)
  -/
  exact Finset.fold_congr hfg
  /-
    🎉 no goals
  -/


theorem gcd_mono_fun {g : β → α} (h : ∀ b ∈ s, f b ∣ g b) : s.gcd f ∣ s.gcd g :=
  dvd_gcd fun b hb ↦ (gcd_dvd hb).trans (h b hb)


theorem gcd_mono (h : s₁ ⊆ s₂) : s₂.gcd f ∣ s₁.gcd f :=
  dvd_gcd fun _ hb ↦ gcd_dvd (h hb)


theorem gcd_image [DecidableEq β] {g : γ → β} (s : Finset γ) :
    (s.image g).gcd f = s.gcd (f ∘ g) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : NormalizedGCDMonoid α
    f : β → α
    inst✝ : DecidableEq β
    g : γ → β
    s : Finset γ
    ⊢ Eq ((Finset.image g s).gcd f) (s.gcd (Function.comp f g))
  -/
  classical induction s using Finset.induction <;> simp [*]
  /-
    🎉 no goals
  -/


theorem gcd_eq_gcd_image [DecidableEq α] : s.gcd f = (s.image f).gcd id :=
  Eq.symm <| gcd_image _


theorem gcd_eq_zero_iff : s.gcd f = 0 ↔ ∀ x : β, x ∈ s → f x = 0 := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    ⊢ Iff (Eq (s.gcd f) 0) (∀ (x : β), Membership.mem s x → Eq (f x) 0)
  -/
  rw [gcd_def, Multiset.gcd_eq_zero_iff]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NormalizedGCDMonoid α
    s : Finset β
    f : β → α
    ⊢ Iff (∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0) (∀ (x : β) …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
      ⊢ ∀ (x : β), Membership.mem s x → Eq (f x) 0
    -/
  · intro b bs
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
      b : β
      bs : Membership.mem s b
      ⊢ Eq (f b) 0
    -/
    apply h (f b)
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
      b : β
      bs : Membership.mem s b
      ⊢ Membership.mem (Multiset.map f s.val) (f b)
    -/
    simp only [Multiset.mem_map, mem_def.1 bs]
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
      b : β
      bs : Membership.mem s b
      ⊢ Exists fun a => And (Membership.mem s.val a) (Eq (f a) (f b))
    -/
    use b
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
      b : β
      bs : Membership.mem s b
      ⊢ And (Membership.mem s.val b) (Eq (f b) (f b))
    -/
    simp only [mem_def.1 bs, eq_self_iff_true, and_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : β), Membership.mem s x → Eq (f x) 0
      ⊢ ∀ (x : α), Membership.mem (Multiset.map f s.val) x → Eq x 0
    -/
  · intro a as
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : β), Membership.mem s x → Eq (f x) 0
      a : α
      as : Membership.mem (Multiset.map f s.val) a
      ⊢ Eq a 0
    -/
    rw [Multiset.mem_map] at as
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : β), Membership.mem s x → Eq (f x) 0
      a : α
      as : Exists fun a_1 => And (Membership.mem s.val a_1) (Eq (f a_1) a)
      ⊢ Eq a 0
    -/
    rcases as with ⟨b, ⟨bs, rfl⟩⟩
    /-
      case mpr.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f : β → α
      h : ∀ (x : β), Membership.mem s x → Eq (f x) 0
      b : β
      bs : Membership.mem s.val b
      ⊢ Eq (f b) 0
    -/
    apply h b (mem_def.1 bs)
    /-
      🎉 no goals
    -/

/- Porting note: The change from `p : α → Prop` to `p : α → Bool` made this slightly less nice with
all the `decide`s around. -/

theorem gcd_eq_gcd_filter_ne_zero [DecidablePred fun x : β ↦ f x = 0] :
    s.gcd f = (s.filter fun x ↦ f x ≠ 0).gcd f := by
  classical
    trans ((s.filter fun x ↦ f x = 0) ∪ s.filter fun x ↦ (f x ≠ 0)).gcd f
    · rw [filter_union_filter_neg_eq]
    rw [gcd_union]
    refine Eq.trans (?_ : _ = GCDMonoid.gcd (0 : α) ?_) (?_ : GCDMonoid.gcd (0 : α) _ = _)
    · exact (gcd (filter (fun x => (f x ≠ 0)) s) f)
    · refine congr (congr rfl <| s.induction_on ?_ ?_) (by simp)
      · simp
      · intro a s _ h
        rw [filter_insert]
        split_ifs with h1 <;> simp [h, h1]
    simp only [gcd_zero_left, normalize_gcd]


nonrec theorem gcd_mul_left {a : α} : (s.gcd fun x ↦ a * f x) = normalize a * s.gcd f := by
  classical
    refine s.induction_on ?_ ?_
    · simp
    · intro b t _ h
      rw [gcd_insert, gcd_insert, h, ← gcd_mul_left]
      apply ((normalize_associated a).mul_right _).gcd_eq_right


nonrec theorem gcd_mul_right {a : α} : (s.gcd fun x ↦ f x * a) = s.gcd f * normalize a := by
  classical
    refine s.induction_on ?_ ?_
    · simp
    · intro b t _ h
      rw [gcd_insert, gcd_insert, h, ← gcd_mul_right]
      apply ((normalize_associated a).mul_left _).gcd_eq_right


theorem extract_gcd' (f g : β → α) (hs : ∃ x, x ∈ s ∧ f x ≠ 0)
    (hg : ∀ b ∈ s, f b = s.gcd f * g b) : s.gcd g = 1 :=
  ((@mul_right_eq_self₀ _ _ (s.gcd f) _).1 <| by
        /-
          α : Type u_2
          β : Type u_3
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NormalizedGCDMonoid α
          s : Finset β
          f g : β → α
          hs : Exists fun x => And (Membership.mem s x) (Ne (f x) 0)
          hg : ∀ (b : β), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
          ⊢ Eq (HMul.hMul (s.gcd f) (s.gcd g)) (s.gcd f)
        -/
        conv_lhs => rw [← normalize_gcd, ← gcd_mul_left, ← gcd_congr rfl hg]).resolve_right <| by
        /-
          🎉 no goals
        -/
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f g : β → α
      hs : Exists fun x => And (Membership.mem s x) (Ne (f x) 0)
      hg : ∀ (b : β), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
      ⊢ Not (Eq (s.gcd f) 0)
    -/
    contrapose! hs
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : NormalizedGCDMonoid α
      s : Finset β
      f g : β → α
      hg : ∀ (b : β), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
      hs : Eq (s.gcd f) 0
      ⊢ ∀ (x : β), Membership.mem s x → Eq (f x) 0
    -/
    exact gcd_eq_zero_iff.1 hs
    /-
      🎉 no goals
    -/


theorem extract_gcd (f : β → α) (hs : s.Nonempty) :
    ∃ g : β → α, (∀ b ∈ s, f b = s.gcd f * g b) ∧ s.gcd g = 1 := by
  classical
    by_cases h : ∀ x ∈ s, f x = (0 : α)
    · refine ⟨fun _ ↦ 1, fun b hb ↦ by rw [h b hb, gcd_eq_zero_iff.2 h, mul_one], ?_⟩
      rw [gcd_eq_gcd_image, image_const hs, gcd_singleton, id, normalize_one]
    · choose g' hg using @gcd_dvd _ _ _ _ s f
      push_neg at h
      refine ⟨fun b ↦ if hb : b ∈ s then g' hb else 0, fun b hb ↦ ?_,
          extract_gcd' f _ h fun b hb ↦ ?_⟩
      · simp only [hb, hg, dite_true]
      rw [dif_pos hb, hg hb]


/-- Given a nonempty Finset `s` and a function `f` from `s` to `ℕ`, if `d = s.gcd`,
then the `gcd` of `(f i) / d` is equal to `1`. -/
lemma gcd_div_eq_one (his : i ∈ s) (hfi : f i ≠ 0) : s.gcd (fun j ↦ f j / s.gcd f) = 1 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizedGCDMonoid α
    inst✝¹ : Div α
    inst✝ : MulDivCancelClass α
    f : ι → α
    s : Finset ι
    i : ι
    his : Membership.mem s i
    hfi : Ne (f i) 0
    ⊢ Eq (s.gcd fun j => HDiv.hDiv (f j) (s.gcd f)) 1
  -/
  obtain ⟨g, he, hg⟩ := Finset.extract_gcd f ⟨i, his⟩
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizedGCDMonoid α
    inst✝¹ : Div α
    inst✝ : MulDivCancelClass α
    f : ι → α
    s : Finset ι
    i : ι
    his : Membership.mem s i
    hfi : Ne (f i) 0
    g : ι → α
    he : ∀ (b : ι), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
    hg : Eq (s.gcd g) 1
    ⊢ Eq (s.gcd fun j => HDiv.hDiv (f j) (s.gcd f)) 1
  -/
  refine (Finset.gcd_congr rfl fun a ha ↦ ?_).trans hg
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizedGCDMonoid α
    inst✝¹ : Div α
    inst✝ : MulDivCancelClass α
    f : ι → α
    s : Finset ι
    i : ι
    his : Membership.mem s i
    hfi : Ne (f i) 0
    g : ι → α
    he : ∀ (b : ι), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
    hg : Eq (s.gcd g) 1
    a : ι
    ha : Membership.mem s a
    ⊢ Eq (HDiv.hDiv (f a) (s.gcd f)) (g a)
  -/
  rw [he a ha, mul_div_cancel_left₀]
  /-
    case intro.intro.ha
    ι : Type u_1
    α : Type u_2
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : NormalizedGCDMonoid α
    inst✝¹ : Div α
    inst✝ : MulDivCancelClass α
    f : ι → α
    s : Finset ι
    i : ι
    his : Membership.mem s i
    hfi : Ne (f i) 0
    g : ι → α
    he : ∀ (b : ι), Membership.mem s b → Eq (f b) (HMul.hMul (s.gcd f) (g b))
    hg : Eq (s.gcd g) 1
    a : ι
    ha : Membership.mem s a
    ⊢ Ne (s.gcd f) 0
  -/
  exact mt Finset.gcd_eq_zero_iff.1 fun h ↦ hfi <| h i his
  /-
    🎉 no goals
  -/


lemma gcd_div_id_eq_one {s : Finset α} {a : α} (has : a ∈ s) (ha : a ≠ 0) :
    s.gcd (fun b ↦ b / s.gcd id) = 1 := gcd_div_eq_one has ha


theorem gcd_eq_of_dvd_sub {s : Finset β} {f g : β → α} {a : α}
    (h : ∀ x : β, x ∈ s → a ∣ f x - g x) :
    GCDMonoid.gcd a (s.gcd f) = GCDMonoid.gcd a (s.gcd g) := by
  classical
    revert h
    refine s.induction_on ?_ ?_
    · simp
    intro b s _ hi h
    rw [gcd_insert, gcd_insert, gcd_comm (f b), ← gcd_assoc,
      hi fun x hx ↦ h _ (mem_insert_of_mem hx), gcd_comm a, gcd_assoc,
      gcd_comm a (GCDMonoid.gcd _ _), gcd_comm (g b), gcd_assoc _ _ a, gcd_comm _ a]
    exact congr_arg _ (gcd_eq_of_dvd_sub_right (h _ (mem_insert_self _ _)))



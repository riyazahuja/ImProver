/-- `Finsupp α M`, denoted `α →₀ M`, is the type of functions `f : α → M` such that
  `f x = 0` for all but finitely many `x`. -/
structure Finsupp (α : Type*) (M : Type*) [Zero M] where
  /-- The support of a finitely supported function (aka `Finsupp`). -/
  support : Finset α
  /-- The underlying function of a bundled finitely supported function (aka `Finsupp`). -/
  toFun : α → M
  /-- The witness that the support of a `Finsupp` is indeed the exact locus where its
  underlying function is nonzero. -/
  mem_support_toFun : ∀ a, a ∈ support ↔ toFun a ≠ 0


@[inherit_doc]
infixr:25 " →₀ " => Finsupp


instance instFunLike : FunLike (α →₀ M) α M :=
  ⟨toFun, by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      ⊢ Function.Injective Finsupp.toFun
    -/
    rintro ⟨s, f, hf⟩ ⟨t, g, hg⟩ (rfl : f = g)
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      s : Finset α
      f : α → M
      hf : ∀ (a : α), Iff (Membership.mem s a) (Ne (f a) 0)
      t : Finset α
      hg : ∀ (a : α), Iff (Membership.mem t a) (Ne (f a) 0)
      ⊢ Eq { support := s, toFun := f, mem_support_toFun := hf } { support := t, toF …
    -/
    congr
    /-
      case mk.mk.e_support
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      s : Finset α
      f : α → M
      hf : ∀ (a : α), Iff (Membership.mem s a) (Ne (f a) 0)
      t : Finset α
      hg : ∀ (a : α), Iff (Membership.mem t a) (Ne (f a) 0)
      ⊢ Eq s t
    -/
    ext a
    /-
      case mk.mk.e_support.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      s : Finset α
      f : α → M
      hf : ∀ (a : α), Iff (Membership.mem s a) (Ne (f a) 0)
      t : Finset α
      hg : ∀ (a : α), Iff (Membership.mem t a) (Ne (f a) 0)
      a : α
      ⊢ Iff (Membership.mem s a) (Membership.mem t a)
    -/
    exact (hf _).trans (hg _).symm⟩
    /-
      🎉 no goals
    -/


@[ext]
theorem ext {f g : α →₀ M} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext _ _ h


lemma ne_iff {f g : α →₀ M} : f ≠ g ↔ ∃ a, f a ≠ g a := DFunLike.ne_iff


@[simp, norm_cast]
theorem coe_mk (f : α → M) (s : Finset α) (h : ∀ a, a ∈ s ↔ f a ≠ 0) : ⇑(⟨s, f, h⟩ : α →₀ M) = f :=
  rfl


instance instZero : Zero (α →₀ M) :=
  ⟨⟨∅, 0, fun _ => ⟨fun h ↦ (not_mem_empty _ h).elim, fun H => (H rfl).elim⟩⟩⟩


@[simp, norm_cast] lemma coe_zero : ⇑(0 : α →₀ M) = 0 := rfl


theorem zero_apply {a : α} : (0 : α →₀ M) a = 0 :=
  rfl


@[simp]
theorem support_zero : (0 : α →₀ M).support = ∅ :=
  rfl


instance instInhabited : Inhabited (α →₀ M) :=
  ⟨0⟩


@[simp]
theorem mem_support_iff {f : α →₀ M} : ∀ {a : α}, a ∈ f.support ↔ f a ≠ 0 :=
  @(f.mem_support_toFun)


@[simp, norm_cast]
theorem fun_support_eq (f : α →₀ M) : Function.support f = f.support :=
  Set.ext fun _x => mem_support_iff.symm


theorem not_mem_support_iff {f : α →₀ M} {a} : a ∉ f.support ↔ f a = 0 :=
  not_iff_comm.1 mem_support_iff.symm


@[simp, norm_cast]
                                                                 /-
                                                                   α : Type u_1
                                                                   M : Type u_5
                                                                   inst✝ : Zero M
                                                                   f : Finsupp α M
                                                                   ⊢ Iff (Eq (⇑f) 0) (Eq f 0)
                                                                 -/
theorem coe_eq_zero {f : α →₀ M} : (f : α → M) = 0 ↔ f = 0 := by rw [← coe_zero, DFunLike.coe_fn_eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem ext_iff' {f g : α →₀ M} : f = g ↔ f.support = g.support ∧ ∀ x ∈ f.support, f x = g x :=
  ⟨fun h => h ▸ ⟨rfl, fun _ _ => rfl⟩, fun ⟨h₁, h₂⟩ =>
    ext fun a => by
      classical
      exact if h : a ∈ f.support then h₂ a h else by
        have hf : f a = 0 := not_mem_support_iff.1 h
        have hg : g a = 0 := by rwa [h₁, not_mem_support_iff] at h
        rw [hf, hg]⟩


@[simp]
theorem support_eq_empty {f : α →₀ M} : f.support = ∅ ↔ f = 0 :=
  mod_cast @Function.support_eq_empty_iff _ _ _ f


theorem support_nonempty_iff {f : α →₀ M} : f.support.Nonempty ↔ f ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    ⊢ Iff f.support.Nonempty (Ne f 0)
  -/
  simp only [Finsupp.support_eq_empty, Finset.nonempty_iff_ne_empty, Ne]
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           α : Type u_1
                                                                           M : Type u_5
                                                                           inst✝ : Zero M
                                                                           f : Finsupp α M
                                                                           ⊢ Iff (Eq f.support.card 0) (Eq f 0)
                                                                         -/
theorem card_support_eq_zero {f : α →₀ M} : #f.support = 0 ↔ f = 0 := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance instDecidableEq [DecidableEq α] [DecidableEq M] : DecidableEq (α →₀ M) := fun f g =>
  decidable_of_iff (f.support = g.support ∧ ∀ a ∈ f.support, f a = g a) ext_iff'.symm


theorem finite_support (f : α →₀ M) : Set.Finite (Function.support f) :=
  f.fun_support_eq.symm ▸ f.support.finite_toSet


theorem support_subset_iff {s : Set α} {f : α →₀ M} :
    ↑f.support ⊆ s ↔ ∀ a ∉ s, f a = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    s : Set α
    f : Finsupp α M
    ⊢ Iff (HasSubset.Subset (↑f.support) s) (∀ (a : α), Not (Membership.mem s a) → …
  -/
  simp only [Set.subset_def, mem_coe, mem_support_iff]; exact forall_congr' fun a => not_imp_comm
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Given `Finite α`, `equivFunOnFinite` is the `Equiv` between `α →₀ β` and `α → β`.
  (All functions on a finite type are finitely supported.) -/
@[simps]
def equivFunOnFinite [Finite α] : (α →₀ M) ≃ (α → M) where
  toFun := (⇑)
  invFun f := mk (Function.support f).toFinite.toFinset f fun _a => Set.Finite.mem_toFinset _
  left_inv _f := ext fun _x => rfl
  right_inv _f := rfl


@[simp]
theorem equivFunOnFinite_symm_coe {α} [Finite α] (f : α →₀ M) : equivFunOnFinite.symm f = f :=
  equivFunOnFinite.symm_apply_apply f


/--
If `α` has a unique term, the type of finitely supported functions `α →₀ β` is equivalent to `β`.
-/
@[simps!]
noncomputable def _root_.Equiv.finsuppUnique {ι : Type*} [Unique ι] : (ι →₀ M) ≃ M :=
  Finsupp.equivFunOnFinite.trans (Equiv.funUnique ι M)


@[ext]
theorem unique_ext [Unique α] {f g : α →₀ M} (h : f default = g default) : f = g :=
                  /-
                    α : Type u_1
                    M : Type u_5
                    inst✝¹ : Zero M
                    inst✝ : Unique α
                    f g : Finsupp α M
                    h : Eq (f Inhabited.default) (g Inhabited.default)
                    a : α
                    ⊢ Eq (f a) (g a)
                  -/
  ext fun a => by rwa [Unique.eq_default a]
                  /-
                    🎉 no goals
                  -/


/-- `single a b` is the finitely supported function with value `b` at `a` and zero otherwise. -/
def single (a : α) (b : M) : α →₀ M where
  support :=
    haveI := Classical.decEq M
    if b = 0 then ∅ else {a}
  toFun :=
    haveI := Classical.decEq α
    Pi.single a b
  mem_support_toFun a' := by
    classical
      obtain rfl | hb := eq_or_ne b 0
      · simp [Pi.single, update]
      rw [if_neg hb, mem_singleton]
      obtain rfl | ha := eq_or_ne a' a
      · simp [hb, Pi.single, update]
      simp [Pi.single_eq_of_ne' ha.symm, ha]


theorem single_apply [Decidable (a = a')] : single a b a' = if a = a' then b else 0 := by
  classical
  simp_rw [@eq_comm _ a a']
  convert Pi.single_apply a b a'


theorem single_apply_left {f : α → β} (hf : Function.Injective f) (x z : α) (y : M) :
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                M : Type u_5
                                                inst✝ : Zero M
                                                f : α → β
                                                hf : Function.Injective f
                                                x z : α
                                                y : M
                                                ⊢ Eq ((Finsupp.single (f x) y) (f z)) ((Finsupp.single x y) z)
                                              -/
    single (f x) y (f z) = single x y z := by classical simp only [single_apply, hf.eq_iff]
                                              /-
                                                🎉 no goals
                                              -/


theorem single_eq_set_indicator : ⇑(single a b) = Set.indicator {a} fun _ => b := by
  classical
  ext
  simp [single_apply, Set.indicator, @eq_comm _ a]


@[simp]
theorem single_eq_same : (single a b : α →₀ M) a = b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b : M
    ⊢ Eq ((Finsupp.single a b) a) b
  -/
  classical exact Pi.single_eq_same (f := fun _ ↦ M) a b
  /-
    🎉 no goals
  -/


@[simp]
theorem single_eq_of_ne (h : a ≠ a') : (single a b : α →₀ M) a' = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a a' : α
    b : M
    h : Ne a a'
    ⊢ Eq ((Finsupp.single a b) a') 0
  -/
  classical exact Pi.single_eq_of_ne' h _
  /-
    🎉 no goals
  -/


theorem single_eq_update [DecidableEq α] (a : α) (b : M) :
    ⇑(single a b) = Function.update (0 : _) a b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    a : α
    b : M
    ⊢ Eq (⇑(Finsupp.single a b)) (Function.update 0 a b)
  -/
  classical rw [single_eq_set_indicator, ← Set.piecewise_eq_indicator, Set.piecewise_singleton]
  /-
    🎉 no goals
  -/


theorem single_eq_pi_single [DecidableEq α] (a : α) (b : M) : ⇑(single a b) = Pi.single a b :=
  single_eq_update a b


@[simp]
theorem single_zero (a : α) : (single a 0 : α →₀ M) = 0 :=
  DFunLike.coe_injective <| by
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      ⊢ Eq ((fun f => ⇑f) (Finsupp.single a 0)) ((fun f => ⇑f) 0)
    -/
    classical simpa only [single_eq_update, coe_zero] using Function.update_eq_self a (0 : α → M)
    /-
      🎉 no goals
    -/


theorem single_of_single_apply (a a' : α) (b : M) :
    single a ((single a' b) a) = single a' (single a' b) a := by
  classical
  rw [single_apply, single_apply]
  ext
  split_ifs with h
  · rw [h]
  · rw [zero_apply, single_apply, ite_self]


theorem support_single_ne_zero (a : α) (hb : b ≠ 0) : (single a b).support = {a} :=
  if_neg hb


theorem support_single_subset : (single a b).support ⊆ {a} := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b : M
    ⊢ HasSubset.Subset (Finsupp.single a b).support (Singleton.singleton a)
  -/
  classical show ite _ _ _ ⊆ _; split_ifs <;> [exact empty_subset _; exact Subset.refl _]
  /-
    🎉 no goals
  -/


theorem single_apply_mem (x) : single a b x ∈ ({0, b} : Set M) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b : M
    x : α
    ⊢ Membership.mem (Insert.insert 0 (Singleton.singleton b)) ((Finsupp.single a  …
  -/
  rcases em (a = x) with (rfl | hx) <;> [simp; simp [single_eq_of_ne hx]]
  /-
    🎉 no goals
  -/


theorem range_single_subset : Set.range (single a b) ⊆ {0, b} :=
  Set.range_subset_iff.2 single_apply_mem


/-- `Finsupp.single a b` is injective in `b`. For the statement that it is injective in `a`, see
`Finsupp.single_left_injective` -/
theorem single_injective (a : α) : Function.Injective (single a : M → α →₀ M) := fun b₁ b₂ eq => by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b₁ b₂ : M
    eq : Eq (Finsupp.single a b₁) (Finsupp.single a b₂)
    ⊢ Eq b₁ b₂
  -/
  have : (single a b₁ : α →₀ M) a = (single a b₂ : α →₀ M) a := by rw [eq]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b₁ b₂ : M
    eq : Eq (Finsupp.single a b₁) (Finsupp.single a b₂)
    this : Eq ((Finsupp.single a b₁) a) ((Finsupp.single a b₂) a)
    ⊢ Eq b₁ b₂
  -/
  rwa [single_eq_same, single_eq_same] at this
  /-
    🎉 no goals
  -/


theorem single_apply_eq_zero {a x : α} {b : M} : single a b x = 0 ↔ x = a → b = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a x : α
    b : M
    ⊢ Iff (Eq ((Finsupp.single a b) x) 0) (Eq x a → Eq b 0)
  -/
  simp [single_eq_set_indicator]
  /-
    🎉 no goals
  -/


theorem single_apply_ne_zero {a x : α} {b : M} : single a b x ≠ 0 ↔ x = a ∧ b ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a x : α
    b : M
    ⊢ Iff (Ne ((Finsupp.single a b) x) 0) (And (Eq x a) (Ne b 0))
  -/
  simp [single_apply_eq_zero]
  /-
    🎉 no goals
  -/


theorem mem_support_single (a a' : α) (b : M) : a ∈ (single a' b).support ↔ a = a' ∧ b ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a a' : α
    b : M
    ⊢ Iff (Membership.mem (Finsupp.single a' b).support a) (And (Eq a a') (Ne b 0))
  -/
  simp [single_apply_eq_zero, not_or]
  /-
    🎉 no goals
  -/


theorem eq_single_iff {f : α →₀ M} {a b} : f = single a b ↔ f.support ⊆ {a} ∧ f a = b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    b : M
    ⊢ Iff (Eq f (Finsupp.single a b)) (And (HasSubset.Subset f.support (Singleton. …
  -/
  refine ⟨fun h => h.symm ▸ ⟨support_single_subset, single_eq_same⟩, ?_⟩
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    b : M
    ⊢ And (HasSubset.Subset f.support (Singleton.singleton a)) (Eq (f a) b) → Eq f …
  -/
  rintro ⟨h, rfl⟩
  /-
    case intro
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    h : HasSubset.Subset f.support (Singleton.singleton a)
    ⊢ Eq f (Finsupp.single a (f a))
  -/
  ext x
  /-
    case intro.h
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    h : HasSubset.Subset f.support (Singleton.singleton a)
    x : α
    ⊢ Eq (f x) ((Finsupp.single a (f a)) x)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : a = x <;> simp only [hx, single_eq_same, single_eq_of_ne, Ne, not_false_iff]
  /-
    case neg
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    h : HasSubset.Subset f.support (Singleton.singleton a)
    x : α
    hx : Not (Eq a x)
    ⊢ Eq (f x) 0
  -/
  exact not_mem_support_iff.1 (mt (fun hx => (mem_singleton.1 (h hx)).symm) hx)
  /-
    🎉 no goals
  -/


theorem single_eq_single_iff (a₁ a₂ : α) (b₁ b₂ : M) :
    single a₁ b₁ = single a₂ b₂ ↔ a₁ = a₂ ∧ b₁ = b₂ ∨ b₁ = 0 ∧ b₂ = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a₁ a₂ : α
    b₁ b₂ : M
    ⊢ Iff (Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂)) (Or (And (Eq a₁ a₂) ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a₁ a₂ : α
      b₁ b₂ : M
      ⊢ Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂) → Or (And (Eq a₁ a₂) (Eq b₁ …
    -/
  · intro eq
    /-
      case mp
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a₁ a₂ : α
      b₁ b₂ : M
      eq : Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂)
      ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
    -/
    by_cases h : a₁ = a₂
      /-
        case pos
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂)
        h : Eq a₁ a₂
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
    · refine Or.inl ⟨h, ?_⟩
      /-
        case pos
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂)
        h : Eq a₁ a₂
        ⊢ Eq b₁ b₂
      -/
      rwa [h, (single_injective a₂).eq_iff] at eq
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : Eq (Finsupp.single a₁ b₁) (Finsupp.single a₂ b₂)
        h : Not (Eq a₁ a₂)
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
    · rw [DFunLike.ext_iff] at eq
      /-
        case neg
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : ∀ (x : α), Eq ((Finsupp.single a₁ b₁) x) ((Finsupp.single a₂ b₂) x)
        h : Not (Eq a₁ a₂)
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
      have h₁ := eq a₁
      /-
        case neg
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : ∀ (x : α), Eq ((Finsupp.single a₁ b₁) x) ((Finsupp.single a₂ b₂) x)
        h : Not (Eq a₁ a₂)
        h₁ : Eq ((Finsupp.single a₁ b₁) a₁) ((Finsupp.single a₂ b₂) a₁)
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
      have h₂ := eq a₂
      /-
        case neg
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : ∀ (x : α), Eq ((Finsupp.single a₁ b₁) x) ((Finsupp.single a₂ b₂) x)
        h : Not (Eq a₁ a₂)
        h₁ : Eq ((Finsupp.single a₁ b₁) a₁) ((Finsupp.single a₂ b₂) a₁)
        h₂ : Eq ((Finsupp.single a₁ b₁) a₂) ((Finsupp.single a₂ b₂) a₂)
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
      simp only [single_eq_same, single_eq_of_ne h, single_eq_of_ne (Ne.symm h)] at h₁ h₂
      /-
        case neg
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        b₁ b₂ : M
        eq : ∀ (x : α), Eq ((Finsupp.single a₁ b₁) x) ((Finsupp.single a₂ b₂) x)
        h : Not (Eq a₁ a₂)
        h₁ : Eq b₁ 0
        h₂ : Eq 0 b₂
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0))
      -/
      exact Or.inr ⟨h₁, h₂.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a₁ a₂ : α
      b₁ b₂ : M
      ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Eq b₁ 0) (Eq b₂ 0)) → Eq (Finsupp.singl …
    -/
  · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
      /-
        case mpr.inl.intro
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ : α
        b₁ : M
        ⊢ Eq (Finsupp.single a₁ b₁) (Finsupp.single a₁ b₁)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        α : Type u_1
        M : Type u_5
        inst✝ : Zero M
        a₁ a₂ : α
        ⊢ Eq (Finsupp.single a₁ 0) (Finsupp.single a₂ 0)
      -/
    · rw [single_zero, single_zero]
      /-
        🎉 no goals
      -/


/-- `Finsupp.single a b` is injective in `a`. For the statement that it is injective in `b`, see
`Finsupp.single_injective` -/
theorem single_left_injective (h : b ≠ 0) : Function.Injective fun a : α => single a b :=
  fun _a _a' H => (((single_eq_single_iff _ _ _ _).mp H).resolve_right fun hb => h hb.1).left


theorem single_left_inj (h : b ≠ 0) : single a b = single a' b ↔ a = a' :=
  (single_left_injective h).eq_iff


theorem support_single_ne_bot (i : α) (h : b ≠ 0) : (single i b).support ≠ ⊥ := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    b : M
    i : α
    h : Ne b 0
    ⊢ Ne (Finsupp.single i b).support Bot.bot
  -/
  simpa only [support_single_ne_zero _ h] using singleton_ne_empty _
  /-
    🎉 no goals
  -/


theorem support_single_disjoint {b' : M} (hb : b ≠ 0) (hb' : b' ≠ 0) {i j : α} :
    Disjoint (single i b).support (single j b').support ↔ i ≠ j := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    b b' : M
    hb : Ne b 0
    hb' : Ne b' 0
    i j : α
    ⊢ Iff (Disjoint (Finsupp.single i b).support (Finsupp.single j b').support) (N …
  -/
  rw [support_single_ne_zero _ hb, support_single_ne_zero _ hb', disjoint_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem single_eq_zero : single a b = 0 ↔ b = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b : M
    ⊢ Iff (Eq (Finsupp.single a b) 0) (Eq b 0)
  -/
  simp [DFunLike.ext_iff, single_eq_set_indicator]
  /-
    🎉 no goals
  -/


theorem single_swap (a₁ a₂ : α) (b : M) : single a₁ b a₂ = single a₂ b a₁ := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a₁ a₂ : α
    b : M
    ⊢ Eq ((Finsupp.single a₁ b) a₂) ((Finsupp.single a₂ b) a₁)
  -/
  classical simp only [single_apply, eq_comm]
  /-
    🎉 no goals
  -/


instance instNontrivial [Nonempty α] [Nontrivial M] : Nontrivial (α →₀ M) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    M : Type u_5
    M' : Type u_6
    N : Type u_7
    P : Type u_8
    G : Type u_9
    H : Type u_10
    R : Type u_11
    S : Type u_12
    inst✝² : Zero M
    a a' : α
    b : M
    inst✝¹ : Nonempty α
    inst✝ : Nontrivial M
    ⊢ Nontrivial (Finsupp α M)
  -/
  inhabit α
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    M : Type u_5
    M' : Type u_6
    N : Type u_7
    P : Type u_8
    G : Type u_9
    H : Type u_10
    R : Type u_11
    S : Type u_12
    inst✝² : Zero M
    a a' : α
    b : M
    inst✝¹ : Nonempty α
    inst✝ : Nontrivial M
    inhabited_h : Inhabited α
    ⊢ Nontrivial (Finsupp α M)
  -/
  rcases exists_ne (0 : M) with ⟨x, hx⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    M : Type u_5
    M' : Type u_6
    N : Type u_7
    P : Type u_8
    G : Type u_9
    H : Type u_10
    R : Type u_11
    S : Type u_12
    inst✝² : Zero M
    a a' : α
    b : M
    inst✝¹ : Nonempty α
    inst✝ : Nontrivial M
    inhabited_h : Inhabited α
    x : M
    hx : Ne x 0
    ⊢ Nontrivial (Finsupp α M)
  -/
  exact nontrivial_of_ne (single default x) 0 (mt single_eq_zero.1 hx)
  /-
    🎉 no goals
  -/


theorem unique_single [Unique α] (x : α →₀ M) : x = single default (x default) :=
  ext <| Unique.forall_iff.2 single_eq_same.symm


@[simp]
theorem unique_single_eq_iff [Unique α] {b' : M} : single a b = single a' b' ↔ b = b' := by
  rw [Finsupp.unique_ext_iff, Unique.eq_default a, Unique.eq_default a', single_eq_same,
    single_eq_same]


lemma apply_single' [Zero N] [Zero P] (e : N → P) (he : e 0 = 0) (a : α) (n : N) (b : α) :
    e ((single a n) b) = single a (e n) b := by
  classical
  simp only [single_apply]
  split_ifs
  · rfl
  · exact he


lemma apply_single [Zero N] [Zero P] {F : Type*} [FunLike F N P] [ZeroHomClass F N P]
    (e : F) (a : α) (n : N) (b : α) :
    e ((single a n) b) = single a (e n) b :=
  apply_single' e (map_zero e) a n b


theorem support_eq_singleton {f : α →₀ M} {a : α} :
    f.support = {a} ↔ f a ≠ 0 ∧ f = single a (f a) :=
  ⟨fun h =>
    ⟨mem_support_iff.1 <| h.symm ▸ Finset.mem_singleton_self a,
      eq_single_iff.2 ⟨subset_of_eq h, rfl⟩⟩,
    fun h => h.2.symm ▸ support_single_ne_zero _ h.1⟩


theorem support_eq_singleton' {f : α →₀ M} {a : α} :
    f.support = {a} ↔ ∃ b ≠ 0, f = single a b :=
  ⟨fun h =>
    let h := support_eq_singleton.1 h
    ⟨_, h.1, h.2⟩,
    fun ⟨_b, hb, hf⟩ => hf.symm ▸ support_single_ne_zero _ hb⟩


theorem card_support_eq_one {f : α →₀ M} :
    #f.support = 1 ↔ ∃ a, f a ≠ 0 ∧ f = single a (f a) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    ⊢ Iff (Eq f.support.card 1) (Exists fun a => And (Ne (f a) 0) (Eq f (Finsupp.s …
  -/
  simp only [card_eq_one, support_eq_singleton]
  /-
    🎉 no goals
  -/


theorem card_support_eq_one' {f : α →₀ M} :
    #f.support = 1 ↔ ∃ a, ∃ b ≠ 0, f = single a b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    ⊢ Iff (Eq f.support.card 1) (Exists fun a => Exists fun b => And (Ne b 0) (Eq  …
  -/
  simp only [card_eq_one, support_eq_singleton']
  /-
    🎉 no goals
  -/


theorem support_subset_singleton {f : α →₀ M} {a : α} : f.support ⊆ {a} ↔ f = single a (f a) :=
  ⟨fun h => eq_single_iff.mpr ⟨h, rfl⟩, fun h => (eq_single_iff.mp h).left⟩


theorem support_subset_singleton' {f : α →₀ M} {a : α} : f.support ⊆ {a} ↔ ∃ b, f = single a b :=
  ⟨fun h => ⟨f a, support_subset_singleton.mp h⟩, fun ⟨b, hb⟩ => by
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      f : Finsupp α M
      a : α
      x✝ : Exists fun b => Eq f (Finsupp.single a b)
      b : M
      hb : Eq f (Finsupp.single a b)
      ⊢ HasSubset.Subset f.support (Singleton.singleton a)
    -/
    rw [hb, support_subset_singleton, single_eq_same]⟩
    /-
      🎉 no goals
    -/


theorem card_support_le_one [Nonempty α] {f : α →₀ M} :
    #f.support ≤ 1 ↔ ∃ a, f = single a (f a) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : Nonempty α
    f : Finsupp α M
    ⊢ Iff (LE.le f.support.card 1) (Exists fun a => Eq f (Finsupp.single a (f a)))
  -/
  simp only [card_le_one_iff_subset_singleton, support_subset_singleton]
  /-
    🎉 no goals
  -/


theorem card_support_le_one' [Nonempty α] {f : α →₀ M} :
    #f.support ≤ 1 ↔ ∃ a b, f = single a b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : Nonempty α
    f : Finsupp α M
    ⊢ Iff (LE.le f.support.card 1) (Exists fun a => Exists fun b => Eq f (Finsupp. …
  -/
  simp only [card_le_one_iff_subset_singleton, support_subset_singleton']
  /-
    🎉 no goals
  -/


@[simp]
theorem equivFunOnFinite_single [DecidableEq α] [Finite α] (x : α) (m : M) :
    Finsupp.equivFunOnFinite (Finsupp.single x m) = Pi.single x m := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : Zero M
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    x : α
    m : M
    ⊢ Eq (Finsupp.equivFunOnFinite (Finsupp.single x m)) (Pi.single x m)
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝² : Zero M
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    x : α
    m : M
    x✝ : α
    ⊢ Eq (Finsupp.equivFunOnFinite (Finsupp.single x m) x✝) (Pi.single x m x✝)
  -/
  simp [Finsupp.single_eq_pi_single, equivFunOnFinite]
  /-
    🎉 no goals
  -/


@[simp]
theorem equivFunOnFinite_symm_single [DecidableEq α] [Finite α] (x : α) (m : M) :
    Finsupp.equivFunOnFinite.symm (Pi.single x m) = Finsupp.single x m := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : Zero M
    inst✝¹ : DecidableEq α
    inst✝ : Finite α
    x : α
    m : M
    ⊢ Eq (Finsupp.equivFunOnFinite.symm (Pi.single x m)) (Finsupp.single x m)
  -/
  rw [← equivFunOnFinite_single, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Replace the value of a `α →₀ M` at a given point `a : α` by a given value `b : M`.
If `b = 0`, this amounts to removing `a` from the `Finsupp.support`.
Otherwise, if `a` was not in the `Finsupp.support`, it is added to it.

This is the finitely-supported version of `Function.update`. -/
def update (f : α →₀ M) (a : α) (b : M) : α →₀ M where
  support := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      f✝ : Finsupp α M
      a✝ : α
      b✝ : M
      i : α
      f : Finsupp α M
      a : α
      b : M
      ⊢ Finset α
    -/
    haveI := Classical.decEq α; haveI := Classical.decEq M
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : Zero M
      f✝ : Finsupp α M
      a✝ : α
      b✝ : M
      i : α
      f : Finsupp α M
      a : α
      b : M
      this✝ : DecidableEq α
      this : DecidableEq M
      ⊢ Finset α
    -/
    exact if b = 0 then f.support.erase a else insert a f.support
    /-
      🎉 no goals
    -/
  toFun :=
    haveI := Classical.decEq α
    Function.update f a b
  mem_support_toFun i := by
    classical
    rw [Function.update]
    simp only [eq_rec_constant, dite_eq_ite, ne_eq]
    split_ifs with hb ha ha <;>
      try simp only [*, not_false_iff, iff_true, not_true, iff_false]
    · rw [Finset.mem_erase]
      simp
    · rw [Finset.mem_erase]
      simp [ha]
    · rw [Finset.mem_insert]
      simp [ha]
    · rw [Finset.mem_insert]
      simp [ha]


@[simp, norm_cast]
theorem coe_update [DecidableEq α] : (f.update a b : α → M) = Function.update f a b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    f : Finsupp α M
    a : α
    b : M
    inst✝ : DecidableEq α
    ⊢ Eq (⇑(f.update a b)) (Function.update (⇑f) a b)
  -/
  delta update Function.update
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    f : Finsupp α M
    a : α
    b : M
    inst✝ : DecidableEq α
    ⊢ Eq ⇑{ support := ite (Eq b 0) (f.support.erase a) (Insert.insert a f.support …
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    f : Finsupp α M
    a : α
    b : M
    inst✝ : DecidableEq α
    x✝ : α
    ⊢ Eq ({ support := ite (Eq b 0) (f.support.erase a) (Insert.insert a f.support …
  -/
  dsimp
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    f : Finsupp α M
    a : α
    b : M
    inst✝ : DecidableEq α
    x✝ : α
    ⊢ Eq (dite (Eq x✝ a) (fun h => Eq.rec b ⋯) fun h => f x✝) (dite (Eq x✝ a) (fun …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem update_self : f.update a (f a) = f := by
  classical
    ext
    simp


@[simp]
theorem zero_update : update 0 a b = single a b := by
  classical
    ext
    rw [single_eq_update]
    rfl


theorem support_update [DecidableEq α] [DecidableEq M] :
    support (f.update a b) = if b = 0 then f.support.erase a else insert a f.support := by
  classical
  dsimp only [update]
  congr!


@[simp]
theorem support_update_zero [DecidableEq α] : support (f.update a 0) = f.support.erase a := by
  classical
  simp only [update, ite_true, mem_support_iff, ne_eq, not_not]
  congr!


theorem support_update_ne_zero [DecidableEq α] (h : b ≠ 0) :
    support (f.update a b) = insert a f.support := by
  classical
  simp only [update, h, ite_false, mem_support_iff, ne_eq]
  congr!


theorem support_update_subset [DecidableEq α] :
    support (f.update a b) ⊆ insert a f.support := by
  classical
  rw [support_update]
  split_ifs
  · exact (erase_subset _ _).trans (subset_insert _ _)
  · rfl


theorem update_comm (f : α →₀ M) {a₁ a₂ : α} (h : a₁ ≠ a₂) (m₁ m₂ : M) :
    update (update f a₁ m₁) a₂ m₂ = update (update f a₂ m₂) a₁ m₁ :=
  letI := Classical.decEq α
  DFunLike.coe_injective <| Function.update_comm h _ _ _


@[simp] theorem update_idem (f : α →₀ M) (a : α) (b c : M) :
    update (update f a b) a c = update f a c :=
  letI := Classical.decEq α
  DFunLike.coe_injective <| Function.update_idem _ _ _


/--
`erase a f` is the finitely supported function equal to `f` except at `a` where it is equal to `0`.
If `a` is not in the support of `f` then `erase a f = f`.
-/
def erase (a : α) (f : α →₀ M) : α →₀ M where
  support :=
    haveI := Classical.decEq α
    f.support.erase a
  toFun a' :=
    haveI := Classical.decEq α
    if a' = a then 0 else f a'
  mem_support_toFun a' := by
    classical
    rw [mem_erase, mem_support_iff]; dsimp
    split_ifs with h
    · exact ⟨fun H _ => H.1 h, fun H => (H rfl).elim⟩
    · exact and_iff_right h


@[simp]
theorem support_erase [DecidableEq α] {a : α} {f : α →₀ M} :
    (f.erase a).support = f.support.erase a := by
  classical
  dsimp only [erase]
  congr!


@[simp]
theorem erase_same {a : α} {f : α →₀ M} : (f.erase a) a = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    f : Finsupp α M
    ⊢ Eq ((Finsupp.erase a f) a) 0
  -/
  classical simp only [erase, coe_mk, ite_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_ne {a a' : α} {f : α →₀ M} (h : a' ≠ a) : (f.erase a) a' = f a' := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a a' : α
    f : Finsupp α M
    h : Ne a' a
    ⊢ Eq ((Finsupp.erase a f) a') (f a')
  -/
  classical simp only [erase, coe_mk, h, ite_false]
  /-
    🎉 no goals
  -/


theorem erase_apply [DecidableEq α] {a a' : α} {f : α →₀ M} :
    f.erase a a' = if a' = a then 0 else f a' := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    a a' : α
    f : Finsupp α M
    ⊢ Eq ((Finsupp.erase a f) a') (ite (Eq a' a) 0 (f a'))
  -/
  rw [erase, coe_mk]
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    a a' : α
    f : Finsupp α M
    ⊢ Eq (ite (Eq a' a) 0 (f a')) (ite (Eq a' a) 0 (f a'))
  -/
  convert rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_single {a : α} {b : M} : erase a (single a b) = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    b : M
    ⊢ Eq (Finsupp.erase a (Finsupp.single a b)) 0
  -/
  ext s; by_cases hs : s = a
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      b : M
      s : α
      hs : Eq s a
      ⊢ Eq ((Finsupp.erase a (Finsupp.single a b)) s) (0 s)
    -/
  · rw [hs, erase_same]
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      b : M
      s : α
      hs : Eq s a
      ⊢ Eq 0 (0 a)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      b : M
      s : α
      hs : Not (Eq s a)
      ⊢ Eq ((Finsupp.erase a (Finsupp.single a b)) s) (0 s)
    -/
  · rw [erase_ne hs]
    /-
      case neg
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a : α
      b : M
      s : α
      hs : Not (Eq s a)
      ⊢ Eq ((Finsupp.single a b) s) (0 s)
    -/
    exact single_eq_of_ne (Ne.symm hs)
    /-
      🎉 no goals
    -/


theorem erase_single_ne {a a' : α} {b : M} (h : a ≠ a') : erase a (single a' b) = single a' b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a a' : α
    b : M
    h : Ne a a'
    ⊢ Eq (Finsupp.erase a (Finsupp.single a' b)) (Finsupp.single a' b)
  -/
  ext s; by_cases hs : s = a
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a a' : α
      b : M
      h : Ne a a'
      s : α
      hs : Eq s a
      ⊢ Eq ((Finsupp.erase a (Finsupp.single a' b)) s) ((Finsupp.single a' b) s)
    -/
  · rw [hs, erase_same, single_eq_of_ne h.symm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      a a' : α
      b : M
      h : Ne a a'
      s : α
      hs : Not (Eq s a)
      ⊢ Eq ((Finsupp.erase a (Finsupp.single a' b)) s) ((Finsupp.single a' b) s)
    -/
  · rw [erase_ne hs]
    /-
      🎉 no goals
    -/


@[simp]
theorem erase_of_not_mem_support {f : α →₀ M} {a} (haf : a ∉ f.support) : erase a f = f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    haf : Not (Membership.mem f.support a)
    ⊢ Eq (Finsupp.erase a f) f
  -/
  ext b; by_cases hab : b = a
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      f : Finsupp α M
      a : α
      haf : Not (Membership.mem f.support a)
      b : α
      hab : Eq b a
      ⊢ Eq ((Finsupp.erase a f) b) (f b)
    -/
  · rwa [hab, erase_same, eq_comm, ← not_mem_support_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      M : Type u_5
      inst✝ : Zero M
      f : Finsupp α M
      a : α
      haf : Not (Membership.mem f.support a)
      b : α
      hab : Not (Eq b a)
      ⊢ Eq ((Finsupp.erase a f) b) (f b)
    -/
  · rw [erase_ne hab]
    /-
      🎉 no goals
    -/


@[simp, nolint simpNF] -- Porting note: simpNF linter claims simp can prove this, it can not
theorem erase_zero (a : α) : erase a (0 : α →₀ M) = 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    a : α
    ⊢ Eq (Finsupp.erase a 0) 0
  -/
  classical rw [← support_eq_empty, support_erase, support_zero, erase_empty]
  /-
    🎉 no goals
  -/


theorem erase_eq_update_zero (f : α →₀ M) (a : α) : f.erase a = update f a 0 :=
  letI := Classical.decEq α
  ext fun _ => (Function.update_apply _ _ _ _).symm

-- The name matches `Finset.erase_insert_of_ne`

theorem erase_update_of_ne (f : α →₀ M) {a a' : α} (ha : a ≠ a') (b : M) :
    erase a (update f a' b) = update (erase a f) a' b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a a' : α
    ha : Ne a a'
    b : M
    ⊢ Eq (Finsupp.erase a (f.update a' b)) ((Finsupp.erase a f).update a' b)
  -/
  rw [erase_eq_update_zero, erase_eq_update_zero, update_comm _ ha]
  /-
    🎉 no goals
  -/

-- not `simp` as `erase_of_not_mem_support` can prove this

theorem erase_idem (f : α →₀ M) (a : α) :
    erase a (erase a f) = erase a f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    ⊢ Eq (Finsupp.erase a (Finsupp.erase a f)) (Finsupp.erase a f)
  -/
  rw [erase_eq_update_zero, erase_eq_update_zero, update_idem]
  /-
    🎉 no goals
  -/


@[simp] theorem update_erase_eq_update (f : α →₀ M) (a : α) (b : M) :
    update (erase a f) a b = update f a b := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    b : M
    ⊢ Eq ((Finsupp.erase a f).update a b) (f.update a b)
  -/
  rw [erase_eq_update_zero, update_idem]
  /-
    🎉 no goals
  -/


@[simp] theorem erase_update_eq_erase (f : α →₀ M) (a : α) (b : M) :
    erase a (update f a b) = erase a f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    f : Finsupp α M
    a : α
    b : M
    ⊢ Eq (Finsupp.erase a (f.update a b)) (Finsupp.erase a f)
  -/
  rw [erase_eq_update_zero, erase_eq_update_zero, update_idem]
  /-
    🎉 no goals
  -/


/-- `Finsupp.onFinset s f hf` is the finsupp function representing `f` restricted to the finset `s`.
The function must be `0` outside of `s`. Use this when the set needs to be filtered anyways,
otherwise a better set representation is often available. -/
def onFinset (s : Finset α) (f : α → M) (hf : ∀ a, f a ≠ 0 → a ∈ s) : α →₀ M where
  support :=
    haveI := Classical.decEq M
    {a ∈ s | f a ≠ 0}
  toFun := f
                          /-
                            α : Type u_1
                            β : Type u_2
                            γ : Type u_3
                            ι : Type u_4
                            M : Type u_5
                            M' : Type u_6
                            N : Type u_7
                            P : Type u_8
                            G : Type u_9
                            H : Type u_10
                            R : Type u_11
                            S : Type u_12
                            inst✝ : Zero M
                            s : Finset α
                            f : α → M
                            hf : ∀ (a : α), Ne (f a) 0 → Membership.mem s a
                            ⊢ ∀ (a : α), Iff (Membership.mem (Finset.filter (fun a => Ne (f a) 0) s) a) (N …
                          -/
  mem_support_toFun := by classical simpa
                          /-
                            🎉 no goals
                          -/


@[simp, norm_cast] lemma coe_onFinset (s : Finset α) (f : α → M) (hf) : onFinset s f hf = f := rfl


@[simp]
theorem onFinset_apply {s : Finset α} {f : α → M} {hf a} : (onFinset s f hf : α →₀ M) a = f a :=
  rfl


@[simp]
theorem support_onFinset_subset {s : Finset α} {f : α → M} {hf} :
    (onFinset s f hf).support ⊆ s := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    s : Finset α
    f : α → M
    hf : ∀ (a : α), Ne (f a) 0 → Membership.mem s a
    ⊢ HasSubset.Subset (Finsupp.onFinset s f hf).support s
  -/
  classical convert filter_subset (f · ≠ 0) s
  /-
    🎉 no goals
  -/


theorem mem_support_onFinset {s : Finset α} {f : α → M} (hf : ∀ a : α, f a ≠ 0 → a ∈ s) {a : α} :
    a ∈ (Finsupp.onFinset s f hf).support ↔ f a ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : Zero M
    s : Finset α
    f : α → M
    hf : ∀ (a : α), Ne (f a) 0 → Membership.mem s a
    a : α
    ⊢ Iff (Membership.mem (Finsupp.onFinset s f hf).support a) (Ne (f a) 0)
  -/
  rw [Finsupp.mem_support_iff, Finsupp.onFinset_apply]
  /-
    🎉 no goals
  -/


theorem support_onFinset [DecidableEq M] {s : Finset α} {f : α → M}
    (hf : ∀ a : α, f a ≠ 0 → a ∈ s) :
    (Finsupp.onFinset s f hf).support = {a ∈ s | f a ≠ 0} := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : Zero M
    inst✝ : DecidableEq M
    s : Finset α
    f : α → M
    hf : ∀ (a : α), Ne (f a) 0 → Membership.mem s a
    ⊢ Eq (Finsupp.onFinset s f hf).support (Finset.filter (fun a => Ne (f a) 0) s)
  -/
  dsimp [onFinset]; congr
                    /-
                      🎉 no goals
                    -/


/-- The natural `Finsupp` induced by the function `f` given that it has finite support. -/
noncomputable def ofSupportFinite (f : α → M) (hf : (Function.support f).Finite) : α →₀ M where
  support := hf.toFinset
  toFun := f
  mem_support_toFun _ := hf.mem_toFinset


theorem ofSupportFinite_coe {f : α → M} {hf : (Function.support f).Finite} :
    (ofSupportFinite f hf : α → M) = f :=
  rfl


instance instCanLift : CanLift (α → M) (α →₀ M) (⇑) fun f => (Function.support f).Finite where
  prf f hf := ⟨ofSupportFinite f hf, rfl⟩


/-- The composition of `f : M → N` and `g : α →₀ M` is `mapRange f hf g : α →₀ N`,
which is well-defined when `f 0 = 0`.

This preserves the structure on `f`, and exists in various bundled forms for when `f` is itself
bundled (defined in `Data/Finsupp/Basic`):

* `Finsupp.mapRange.equiv`
* `Finsupp.mapRange.zeroHom`
* `Finsupp.mapRange.addMonoidHom`
* `Finsupp.mapRange.addEquiv`
* `Finsupp.mapRange.linearMap`
* `Finsupp.mapRange.linearEquiv`
-/
def mapRange (f : M → N) (hf : f 0 = 0) (g : α →₀ M) : α →₀ N :=
  onFinset g.support (f ∘ g) fun a => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝² : Zero M
      inst✝¹ : Zero N
      inst✝ : Zero P
      f : M → N
      hf : Eq (f 0) 0
      g : Finsupp α M
      a : α
      ⊢ Ne (Function.comp f (⇑g) a) 0 → Membership.mem g.support a
    -/
    rw [mem_support_iff, not_imp_not]; exact fun H => (congr_arg f H).trans hf
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem mapRange_apply {f : M → N} {hf : f 0 = 0} {g : α →₀ M} {a : α} :
    mapRange f hf g a = f (g a) :=
  rfl


@[simp]
theorem mapRange_zero {f : M → N} {hf : f 0 = 0} : mapRange f hf (0 : α →₀ M) = 0 :=
                  /-
                    α : Type u_1
                    M : Type u_5
                    N : Type u_7
                    inst✝¹ : Zero M
                    inst✝ : Zero N
                    f : M → N
                    hf : Eq (f 0) 0
                    x✝ : α
                    ⊢ Eq ((Finsupp.mapRange f hf 0) x✝) (0 x✝)
                  -/
  ext fun _ => by simp only [hf, zero_apply, mapRange_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem mapRange_id (g : α →₀ M) : mapRange id rfl g = g :=
  ext fun _ => rfl


theorem mapRange_comp (f : N → P) (hf : f 0 = 0) (f₂ : M → N) (hf₂ : f₂ 0 = 0) (h : (f ∘ f₂) 0 = 0)
    (g : α →₀ M) : mapRange (f ∘ f₂) h g = mapRange f hf (mapRange f₂ hf₂ g) :=
  ext fun _ => rfl


@[simp]
lemma mapRange_mapRange (e₁ : N → P) (e₂ : M → N) (he₁ he₂) (f : α →₀ M) :
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   γ : Type u_3
                                                                   ι : Type u_4
                                                                   M : Type u_5
                                                                   M' : Type u_6
                                                                   N : Type u_7
                                                                   P : Type u_8
                                                                   G : Type u_9
                                                                   H : Type u_10
                                                                   R : Type u_11
                                                                   S : Type u_12
                                                                   inst✝² : Zero M
                                                                   inst✝¹ : Zero N
                                                                   inst✝ : Zero P
                                                                   e₁ : N → P
                                                                   e₂ : M → N
                                                                   he₁ : Eq (e₁ 0) 0
                                                                   he₂ : Eq (e₂ 0) 0
                                                                   f : Finsupp α M
                                                                   ⊢ Eq (Function.comp e₁ e₂ 0) 0
                                                                 -/
    mapRange e₁ he₁ (mapRange e₂ he₂ f) = mapRange (e₁ ∘ e₂) (by simp [*]) f := ext fun _ ↦ rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem support_mapRange {f : M → N} {hf : f 0 = 0} {g : α →₀ M} :
    (mapRange f hf g).support ⊆ g.support :=
  support_onFinset_subset


@[simp]
theorem mapRange_single {f : M → N} {hf : f 0 = 0} {a : α} {b : M} :
    mapRange f hf (single a b) = single a (f b) :=
  ext fun a' => by
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : M → N
      hf : Eq (f 0) 0
      a : α
      b : M
      a' : α
      ⊢ Eq ((Finsupp.mapRange f hf (Finsupp.single a b)) a') ((Finsupp.single a (f b …
    -/
    classical simpa only [single_eq_pi_single] using Pi.apply_single _ (fun _ => hf) a _ a'
    /-
      🎉 no goals
    -/


theorem support_mapRange_of_injective {e : M → N} (he0 : e 0 = 0) (f : ι →₀ M)
    (he : Function.Injective e) : (Finsupp.mapRange e he0 f).support = f.support := by
  /-
    ι : Type u_4
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he0 : Eq (e 0) 0
    f : Finsupp ι M
    he : Function.Injective e
    ⊢ Eq (Finsupp.mapRange e he0 f).support f.support
  -/
  ext
  /-
    case h
    ι : Type u_4
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he0 : Eq (e 0) 0
    f : Finsupp ι M
    he : Function.Injective e
    a✝ : ι
    ⊢ Iff (Membership.mem (Finsupp.mapRange e he0 f).support a✝) (Membership.mem f …
  -/
  simp only [Finsupp.mem_support_iff, Ne, Finsupp.mapRange_apply]
  /-
    case h
    ι : Type u_4
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he0 : Eq (e 0) 0
    f : Finsupp ι M
    he : Function.Injective e
    a✝ : ι
    ⊢ Iff (Not (Eq (e (f a✝)) 0)) (Not (Eq (f a✝) 0))
  -/
  exact he.ne_iff' he0
  /-
    🎉 no goals
  -/


lemma range_mapRange (e : M → N) (he₀ : e 0 = 0) :
    Set.range (Finsupp.mapRange (α := α) e he₀) = {g | ∀ i, g i ∈ Set.range e} := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    ⊢ Eq (Set.range (Finsupp.mapRange e he₀)) (setOf fun g => ∀ (i : α), Membershi …
  -/
  ext g
  /-
    case h
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    g : Finsupp α N
    ⊢ Iff (Membership.mem (Set.range (Finsupp.mapRange e he₀)) g) (Membership.mem  …
  -/
  simp only [Set.mem_range, Set.mem_setOf]
  /-
    case h
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    g : Finsupp α N
    ⊢ Iff (Exists fun y => Eq (Finsupp.mapRange e he₀ y) g) (∀ (i : α), Exists fun …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      e : M → N
      he₀ : Eq (e 0) 0
      g : Finsupp α N
      ⊢ (Exists fun y => Eq (Finsupp.mapRange e he₀ y) g) → ∀ (i : α), Exists fun y  …
    -/
  · rintro ⟨g, rfl⟩ i
    /-
      case h.mp.intro
      α : Type u_1
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      e : M → N
      he₀ : Eq (e 0) 0
      g : Finsupp α M
      i : α
      ⊢ Exists fun y => Eq (e y) ((Finsupp.mapRange e he₀ g) i)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      e : M → N
      he₀ : Eq (e 0) 0
      g : Finsupp α N
      ⊢ (∀ (i : α), Exists fun y => Eq (e y) (g i)) → Exists fun y => Eq (Finsupp.ma …
    -/
  · intro h
    classical
    choose f h using h
    use onFinset g.support (Set.indicator g.support f) (by aesop)
    ext i
    simp only [mapRange_apply, onFinset_apply, Set.indicator_apply]
    split_ifs <;> simp_all


/-- `Finsupp.mapRange` of a injective function is injective. -/
lemma mapRange_injective (e : M → N) (he₀ : e 0 = 0) (he : Injective e) :
    Injective (Finsupp.mapRange (α := α) e he₀) := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    he : Function.Injective e
    ⊢ Function.Injective (Finsupp.mapRange e he₀)
  -/
  intro a b h
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    he : Function.Injective e
    a b : Finsupp α M
    h : Eq (Finsupp.mapRange e he₀ a) (Finsupp.mapRange e he₀ b)
    ⊢ Eq a b
  -/
  rw [Finsupp.ext_iff] at h ⊢
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    he : Function.Injective e
    a b : Finsupp α M
    h : ∀ (a_1 : α), Eq ((Finsupp.mapRange e he₀ a) a_1) ((Finsupp.mapRange e he₀  …
    ⊢ ∀ (a_1 : α), Eq (a a_1) (b a_1)
  -/
  simpa only [mapRange_apply, he.eq_iff] using h
  /-
    🎉 no goals
  -/


/-- `Finsupp.mapRange` of a surjective function is surjective. -/
lemma mapRange_surjective (e : M → N) (he₀ : e 0 = 0) (he : Surjective e) :
    Surjective (Finsupp.mapRange (α := α) e he₀) := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    he : Function.Surjective e
    ⊢ Function.Surjective (Finsupp.mapRange e he₀)
  -/
  rw [← Set.range_eq_univ, range_mapRange, he.range_eq]
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    e : M → N
    he₀ : Eq (e 0) 0
    he : Function.Surjective e
    ⊢ Eq (setOf fun g => ∀ (i : α), Membership.mem Set.univ (g i)) Set.univ
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given `f : α ↪ β` and `v : α →₀ M`, `Finsupp.embDomain f v : β →₀ M`
is the finitely supported function whose value at `f a : β` is `v a`.
For a `b : β` outside the range of `f`, it is zero. -/
def embDomain (f : α ↪ β) (v : α →₀ M) : β →₀ M where
  support := v.support.map f
  toFun a₂ :=
    haveI := Classical.decEq β
    if h : a₂ ∈ v.support.map f then
      v
        (v.support.choose (fun a₁ => f a₁ = a₂)
          (by
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              ι : Type u_4
              M : Type u_5
              M' : Type u_6
              N : Type u_7
              P : Type u_8
              G : Type u_9
              H : Type u_10
              R : Type u_11
              S : Type u_12
              inst✝¹ : Zero M
              inst✝ : Zero N
              f : Function.Embedding α β
              v : Finsupp α M
              a₂ : β
              this : DecidableEq β
              h : Membership.mem (Finset.map f v.support) a₂
              ⊢ ExistsUnique fun a => And (Membership.mem v.support a) ((fun a₁ => Eq (f a₁) …
            -/
            rcases Finset.mem_map.1 h with ⟨a, ha, rfl⟩
            /-
              case intro.intro
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              ι : Type u_4
              M : Type u_5
              M' : Type u_6
              N : Type u_7
              P : Type u_8
              G : Type u_9
              H : Type u_10
              R : Type u_11
              S : Type u_12
              inst✝¹ : Zero M
              inst✝ : Zero N
              f : Function.Embedding α β
              v : Finsupp α M
              this : DecidableEq β
              a : α
              ha : Membership.mem v.support a
              h : Membership.mem (Finset.map f v.support) (f a)
              ⊢ ExistsUnique fun a_1 => And (Membership.mem v.support a_1) ((fun a₁ => Eq (f …
            -/
            exact ExistsUnique.intro a ⟨ha, rfl⟩ fun b ⟨_, hb⟩ => f.injective hb))
            /-
              🎉 no goals
            -/
    else 0
  mem_support_toFun a₂ := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : Function.Embedding α β
      v : Finsupp α M
      a₂ : β
      ⊢ Iff (Membership.mem (Finset.map f v.support) a₂) (Ne ((fun a₂ => dite (Membe …
    -/
    dsimp
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : Function.Embedding α β
      v : Finsupp α M
      a₂ : β
      ⊢ Iff (Membership.mem (Finset.map f v.support) a₂) (Not (Eq (dite (Membership. …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : Zero M
        inst✝ : Zero N
        f : Function.Embedding α β
        v : Finsupp α M
        a₂ : β
        h : Membership.mem (Finset.map f v.support) a₂
        ⊢ Iff (Membership.mem (Finset.map f v.support) a₂) (Not (Eq (v (Finset.choose  …
      -/
    · simp only [h, true_iff, Ne]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : Zero M
        inst✝ : Zero N
        f : Function.Embedding α β
        v : Finsupp α M
        a₂ : β
        h : Membership.mem (Finset.map f v.support) a₂
        ⊢ Not (Eq (v (Finset.choose (fun a₁ => Eq (f a₁) a₂) v.support ⋯)) 0)
      -/
      rw [← not_mem_support_iff, not_not]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : Zero M
        inst✝ : Zero N
        f : Function.Embedding α β
        v : Finsupp α M
        a₂ : β
        h : Membership.mem (Finset.map f v.support) a₂
        ⊢ Membership.mem v.support (Finset.choose (fun a₁ => Eq (f a₁) a₂) v.support ⋯)
      -/
      classical apply Finset.choose_mem
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝¹ : Zero M
        inst✝ : Zero N
        f : Function.Embedding α β
        v : Finsupp α M
        a₂ : β
        h : Not (Membership.mem (Finset.map f v.support) a₂)
        ⊢ Iff (Membership.mem (Finset.map f v.support) a₂) (Not (Eq 0 0))
      -/
    · simp only [h, Ne, ne_self_iff_false, not_true_eq_false]
      /-
        🎉 no goals
      -/


@[simp]
theorem support_embDomain (f : α ↪ β) (v : α →₀ M) : (embDomain f v).support = v.support.map f :=
  rfl


@[simp]
theorem embDomain_zero (f : α ↪ β) : (embDomain f 0 : β →₀ M) = 0 :=
  rfl


@[simp]
theorem embDomain_apply (f : α ↪ β) (v : α →₀ M) (a : α) : embDomain f v (f a) = v a := by
  classical
    change dite _ _ _ = _
    split_ifs with h <;> rw [Finset.mem_map' f] at h
    · refine congr_arg (v : α → M) (f.inj' ?_)
      exact Finset.choose_property (fun a₁ => f a₁ = f a) _ _
    · exact (not_mem_support_iff.1 h).symm


theorem embDomain_notin_range (f : α ↪ β) (v : α →₀ M) (a : β) (h : a ∉ Set.range f) :
    embDomain f v a = 0 := by
  classical
    refine dif_neg (mt (fun h => ?_) h)
    rcases Finset.mem_map.1 h with ⟨a, _h, rfl⟩
    exact Set.mem_range_self a


theorem embDomain_injective (f : α ↪ β) : Function.Injective (embDomain f : (α →₀ M) → β →₀ M) :=
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   M : Type u_5
                                   inst✝ : Zero M
                                   f : Function.Embedding α β
                                   l₁ l₂ : Finsupp α M
                                   h : Eq (Finsupp.embDomain f l₁) (Finsupp.embDomain f l₂)
                                   a : α
                                   ⊢ Eq (l₁ a) (l₂ a)
                                 -/
  fun l₁ l₂ h => ext fun a => by simpa only [embDomain_apply] using DFunLike.ext_iff.1 h (f a)
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem embDomain_inj {f : α ↪ β} {l₁ l₂ : α →₀ M} : embDomain f l₁ = embDomain f l₂ ↔ l₁ = l₂ :=
  (embDomain_injective f).eq_iff


@[simp]
theorem embDomain_eq_zero {f : α ↪ β} {l : α →₀ M} : embDomain f l = 0 ↔ l = 0 :=
  (embDomain_injective f).eq_iff' <| embDomain_zero f


theorem embDomain_mapRange (f : α ↪ β) (g : M → N) (p : α →₀ M) (hg : g 0 = 0) :
    embDomain f (mapRange g hg p) = mapRange g hg (embDomain f p) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    f : Function.Embedding α β
    g : M → N
    p : Finsupp α M
    hg : Eq (g 0) 0
    ⊢ Eq (Finsupp.embDomain f (Finsupp.mapRange g hg p)) (Finsupp.mapRange g hg (F …
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    M : Type u_5
    N : Type u_7
    inst✝¹ : Zero M
    inst✝ : Zero N
    f : Function.Embedding α β
    g : M → N
    p : Finsupp α M
    hg : Eq (g 0) 0
    a : β
    ⊢ Eq ((Finsupp.embDomain f (Finsupp.mapRange g hg p)) a) ((Finsupp.mapRange g  …
  -/
  by_cases h : a ∈ Set.range f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : Function.Embedding α β
      g : M → N
      p : Finsupp α M
      hg : Eq (g 0) 0
      a : β
      h : Membership.mem (Set.range ⇑f) a
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.mapRange g hg p)) a) ((Finsupp.mapRange g  …
    -/
  · rcases h with ⟨a', rfl⟩
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : Function.Embedding α β
      g : M → N
      p : Finsupp α M
      hg : Eq (g 0) 0
      a' : α
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.mapRange g hg p)) (f a')) ((Finsupp.mapRan …
    -/
    rw [mapRange_apply, embDomain_apply, embDomain_apply, mapRange_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      M : Type u_5
      N : Type u_7
      inst✝¹ : Zero M
      inst✝ : Zero N
      f : Function.Embedding α β
      g : M → N
      p : Finsupp α M
      hg : Eq (g 0) 0
      a : β
      h : Not (Membership.mem (Set.range ⇑f) a)
      ⊢ Eq ((Finsupp.embDomain f (Finsupp.mapRange g hg p)) a) ((Finsupp.mapRange g  …
    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  · rw [mapRange_apply, embDomain_notin_range, embDomain_notin_range, ← hg] <;> assumption
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem single_of_embDomain_single (l : α →₀ M) (f : α ↪ β) (a : β) (b : M) (hb : b ≠ 0)
    (h : l.embDomain f = single a b) : ∃ x, l = single x b ∧ f x = a := by
  classical
    have h_map_support : Finset.map f l.support = {a} := by
      rw [← support_embDomain, h, support_single_ne_zero _ hb]
    have ha : a ∈ Finset.map f l.support := by simp only [h_map_support, Finset.mem_singleton]
    rcases Finset.mem_map.1 ha with ⟨c, _hc₁, hc₂⟩
    use c
    constructor
    · ext d
      rw [← embDomain_apply f l, h]
      by_cases h_cases : c = d
      · simp only [Eq.symm h_cases, hc₂, single_eq_same]
      · rw [single_apply, single_apply, if_neg, if_neg h_cases]
        by_contra hfd
        exact h_cases (f.injective (hc₂.trans hfd))
    · exact hc₂


@[simp]
theorem embDomain_single (f : α ↪ β) (a : α) (m : M) :
    embDomain f (single a m) = single (f a) m := by
  classical
    ext b
    by_cases h : b ∈ Set.range f
    · rcases h with ⟨a', rfl⟩
      simp [single_apply]
    · simp only [embDomain_notin_range, h, single_apply, not_false_iff]
      rw [if_neg]
      rintro rfl
      simp at h


/-- Given finitely supported functions `g₁ : α →₀ M` and `g₂ : α →₀ N` and function `f : M → N → P`,
`Finsupp.zipWith f hf g₁ g₂` is the finitely supported function `α →₀ P` satisfying
`zipWith f hf g₁ g₂ a = f (g₁ a) (g₂ a)`, which is well-defined when `f 0 0 = 0`. -/
def zipWith (f : M → N → P) (hf : f 0 0 = 0) (g₁ : α →₀ M) (g₂ : α →₀ N) : α →₀ P :=
  onFinset
    (haveI := Classical.decEq α; g₁.support ∪ g₂.support)
    (fun a => f (g₁ a) (g₂ a))
    fun a (H : f _ _ ≠ 0) => by
      classical
      rw [mem_union, mem_support_iff, mem_support_iff, ← not_and_or]
      rintro ⟨h₁, h₂⟩; rw [h₁, h₂] at H; exact H hf


@[simp]
theorem zipWith_apply {f : M → N → P} {hf : f 0 0 = 0} {g₁ : α →₀ M} {g₂ : α →₀ N} {a : α} :
    zipWith f hf g₁ g₂ a = f (g₁ a) (g₂ a) :=
  rfl


theorem support_zipWith [D : DecidableEq α] {f : M → N → P} {hf : f 0 0 = 0} {g₁ : α →₀ M}
    {g₂ : α →₀ N} : (zipWith f hf g₁ g₂).support ⊆ g₁.support ∪ g₂.support := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    P : Type u_8
    inst✝² : Zero M
    inst✝¹ : Zero N
    inst✝ : Zero P
    D : DecidableEq α
    f : M → N → P
    hf : Eq (f 0 0) 0
    g₁ : Finsupp α M
    g₂ : Finsupp α N
    ⊢ HasSubset.Subset (Finsupp.zipWith f hf g₁ g₂).support (Union.union g₁.suppor …
  -/
  convert support_onFinset_subset
  /-
    🎉 no goals
  -/


@[simp]
theorem zipWith_single_single (f : M → N → P) (hf : f 0 0 = 0) (a : α) (m : M) (n : N) :
    zipWith f hf (single a m) (single a n) = single a (f m n) := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    P : Type u_8
    inst✝² : Zero M
    inst✝¹ : Zero N
    inst✝ : Zero P
    f : M → N → P
    hf : Eq (f 0 0) 0
    a : α
    m : M
    n : N
    ⊢ Eq (Finsupp.zipWith f hf (Finsupp.single a m) (Finsupp.single a n)) (Finsupp …
  -/
  ext a'
  /-
    case h
    α : Type u_1
    M : Type u_5
    N : Type u_7
    P : Type u_8
    inst✝² : Zero M
    inst✝¹ : Zero N
    inst✝ : Zero P
    f : M → N → P
    hf : Eq (f 0 0) 0
    a : α
    m : M
    n : N
    a' : α
    ⊢ Eq ((Finsupp.zipWith f hf (Finsupp.single a m) (Finsupp.single a n)) a') ((F …
  -/
  rw [zipWith_apply]
  /-
    case h
    α : Type u_1
    M : Type u_5
    N : Type u_7
    P : Type u_8
    inst✝² : Zero M
    inst✝¹ : Zero N
    inst✝ : Zero P
    f : M → N → P
    hf : Eq (f 0 0) 0
    a : α
    m : M
    n : N
    a' : α
    ⊢ Eq (f ((Finsupp.single a m) a') ((Finsupp.single a n) a')) ((Finsupp.single  …
  -/
  obtain rfl | ha' := eq_or_ne a a'
    /-
      case h.inl
      α : Type u_1
      M : Type u_5
      N : Type u_7
      P : Type u_8
      inst✝² : Zero M
      inst✝¹ : Zero N
      inst✝ : Zero P
      f : M → N → P
      hf : Eq (f 0 0) 0
      a : α
      m : M
      n : N
      ⊢ Eq (f ((Finsupp.single a m) a) ((Finsupp.single a n) a)) ((Finsupp.single a  …
    -/
  · rw [single_eq_same, single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      M : Type u_5
      N : Type u_7
      P : Type u_8
      inst✝² : Zero M
      inst✝¹ : Zero N
      inst✝ : Zero P
      f : M → N → P
      hf : Eq (f 0 0) 0
      a : α
      m : M
      n : N
      a' : α
      ha' : Ne a a'
      ⊢ Eq (f ((Finsupp.single a m) a') ((Finsupp.single a n) a')) ((Finsupp.single  …
    -/
  · rw [single_eq_of_ne ha', single_eq_of_ne ha', single_eq_of_ne ha', hf]
    /-
      🎉 no goals
    -/


instance instAdd : Add (α →₀ M) :=
  ⟨zipWith (· + ·) (add_zero 0)⟩


@[simp, norm_cast] lemma coe_add (f g : α →₀ M) : ⇑(f + g) = f + g := rfl


theorem add_apply (g₁ g₂ : α →₀ M) (a : α) : (g₁ + g₂) a = g₁ a + g₂ a :=
  rfl


theorem support_add [DecidableEq α] {g₁ g₂ : α →₀ M} :
    (g₁ + g₂).support ⊆ g₁.support ∪ g₂.support :=
  support_zipWith


theorem support_add_eq [DecidableEq α] {g₁ g₂ : α →₀ M} (h : Disjoint g₁.support g₂.support) :
    (g₁ + g₂).support = g₁.support ∪ g₂.support :=
  le_antisymm support_zipWith fun a ha =>
    (Finset.mem_union.1 ha).elim
      (fun ha => by
        /-
          α : Type u_1
          M : Type u_5
          inst✝¹ : AddZeroClass M
          inst✝ : DecidableEq α
          g₁ g₂ : Finsupp α M
          h : Disjoint g₁.support g₂.support
          a : α
          ha✝ : Membership.mem (Union.union g₁.support g₂.support) a
          ha : Membership.mem g₁.support a
          ⊢ Membership.mem (HAdd.hAdd g₁ g₂).support a
        -/
        have : a ∉ g₂.support := disjoint_left.1 h ha
        /-
          α : Type u_1
          M : Type u_5
          inst✝¹ : AddZeroClass M
          inst✝ : DecidableEq α
          g₁ g₂ : Finsupp α M
          h : Disjoint g₁.support g₂.support
          a : α
          ha✝ : Membership.mem (Union.union g₁.support g₂.support) a
          ha : Membership.mem g₁.support a
          this : Not (Membership.mem g₂.support a)
          ⊢ Membership.mem (HAdd.hAdd g₁ g₂).support a
        -/
        simp only [mem_support_iff, not_not] at *; simpa only [add_apply, this, add_zero] )
                                                   /-
                                                     🎉 no goals
                                                   -/
      fun ha => by
      /-
        α : Type u_1
        M : Type u_5
        inst✝¹ : AddZeroClass M
        inst✝ : DecidableEq α
        g₁ g₂ : Finsupp α M
        h : Disjoint g₁.support g₂.support
        a : α
        ha✝ : Membership.mem (Union.union g₁.support g₂.support) a
        ha : Membership.mem g₂.support a
        ⊢ Membership.mem (HAdd.hAdd g₁ g₂).support a
      -/
      have : a ∉ g₁.support := disjoint_right.1 h ha
      /-
        α : Type u_1
        M : Type u_5
        inst✝¹ : AddZeroClass M
        inst✝ : DecidableEq α
        g₁ g₂ : Finsupp α M
        h : Disjoint g₁.support g₂.support
        a : α
        ha✝ : Membership.mem (Union.union g₁.support g₂.support) a
        ha : Membership.mem g₂.support a
        this : Not (Membership.mem g₁.support a)
        ⊢ Membership.mem (HAdd.hAdd g₁ g₂).support a
      -/
      simp only [mem_support_iff, not_not] at *; simpa only [add_apply, this, zero_add]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem single_add (a : α) (b₁ b₂ : M) : single a (b₁ + b₂) = single a b₁ + single a b₂ :=
  (zipWith_single_single _ _ _ _ _).symm


theorem support_single_add {a : α} {b : M} {f : α →₀ M} (ha : a ∉ f.support) (hb : b ≠ 0) :
    support (single a b + f) = cons a f.support ha := by
  classical
  have H := support_single_ne_zero a hb
  rw [support_add_eq, H, cons_eq_insert, insert_eq]
  rwa [H, disjoint_singleton_left]


theorem support_add_single {a : α} {b : M} {f : α →₀ M} (ha : a ∉ f.support) (hb : b ≠ 0) :
    support (f + single a b) = cons a f.support ha := by
  classical
  have H := support_single_ne_zero a hb
  rw [support_add_eq, H, union_comm, cons_eq_insert, insert_eq]
  rwa [H, disjoint_singleton_right]


instance instAddZeroClass : AddZeroClass (α →₀ M) :=
  DFunLike.coe_injective.addZeroClass _ coe_zero coe_add


instance instIsLeftCancelAdd [IsLeftCancelAdd M] : IsLeftCancelAdd (α →₀ M) where
  add_left_cancel _ _ _ h := ext fun x => add_left_cancel <| DFunLike.congr_fun h x


/-- When ι is finite and M is an AddMonoid,
  then Finsupp.equivFunOnFinite gives an AddEquiv -/
noncomputable def addEquivFunOnFinite {ι : Type*} [Finite ι] :
    (ι →₀ M) ≃+ (ι → M) where
  __ := Finsupp.equivFunOnFinite
  map_add' _ _ := rfl


/-- AddEquiv between (ι →₀ M) and M, when ι has a unique element -/
noncomputable def _root_.AddEquiv.finsuppUnique {ι : Type*} [Unique ι] :
    (ι →₀ M) ≃+ M where
  __ := Equiv.finsuppUnique
  map_add' _ _ := rfl


lemma _root_.AddEquiv.finsuppUnique_symm {M : Type*} [AddZeroClass M] (d : M) :
    AddEquiv.finsuppUnique.symm d = single () d := by
  /-
    M : Type u_13
    inst✝ : AddZeroClass M
    d : M
    ⊢ Eq (AddEquiv.finsuppUnique.symm d) (Finsupp.single Unit.unit d)
  -/
  rw [Finsupp.unique_single (AddEquiv.finsuppUnique.symm d), Finsupp.unique_single_eq_iff]
  /-
    M : Type u_13
    inst✝ : AddZeroClass M
    d : M
    ⊢ Eq ((AddEquiv.finsuppUnique.symm d) Inhabited.default) d
  -/
  simp [AddEquiv.finsuppUnique]
  /-
    🎉 no goals
  -/


instance instIsRightCancelAdd [IsRightCancelAdd M] : IsRightCancelAdd (α →₀ M) where
  add_right_cancel _ _ _ h := ext fun x => add_right_cancel <| DFunLike.congr_fun h x


instance instIsCancelAdd [IsCancelAdd M] : IsCancelAdd (α →₀ M) where


/-- `Finsupp.single` as an `AddMonoidHom`.

See `Finsupp.lsingle` in `LinearAlgebra/Finsupp` for the stronger version as a linear map. -/
@[simps]
def singleAddHom (a : α) : M →+ α →₀ M where
  toFun := single a
  map_zero' := single_zero a
  map_add' := single_add a


/-- Evaluation of a function `f : α →₀ M` at a point as an additive monoid homomorphism.

See `Finsupp.lapply` in `LinearAlgebra/Finsupp` for the stronger version as a linear map. -/
@[simps apply]
def applyAddHom (a : α) : (α →₀ M) →+ M where
  toFun g := g a
  map_zero' := zero_apply
  map_add' _ _ := add_apply _ _ _


/-- Coercion from a `Finsupp` to a function type is an `AddMonoidHom`. -/
@[simps]
noncomputable def coeFnAddHom : (α →₀ M) →+ α → M where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


theorem update_eq_single_add_erase (f : α →₀ M) (a : α) (b : M) :
    f.update a b = single a b + f.erase a := by
  classical
    ext j
    rcases eq_or_ne a j with (rfl | h)
    · simp
    · simp [Function.update_of_ne h.symm, single_apply, h, erase_ne, h.symm]


theorem update_eq_erase_add_single (f : α →₀ M) (a : α) (b : M) :
    f.update a b = f.erase a + single a b := by
  classical
    ext j
    rcases eq_or_ne a j with (rfl | h)
    · simp
    · simp [Function.update_of_ne h.symm, single_apply, h, erase_ne, h.symm]


theorem single_add_erase (a : α) (f : α →₀ M) : single a (f a) + f.erase a = f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : AddZeroClass M
    a : α
    f : Finsupp α M
    ⊢ Eq (HAdd.hAdd (Finsupp.single a (f a)) (Finsupp.erase a f)) f
  -/
  rw [← update_eq_single_add_erase, update_self]
  /-
    🎉 no goals
  -/


theorem erase_add_single (a : α) (f : α →₀ M) : f.erase a + single a (f a) = f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : AddZeroClass M
    a : α
    f : Finsupp α M
    ⊢ Eq (HAdd.hAdd (Finsupp.erase a f) (Finsupp.single a (f a))) f
  -/
  rw [← update_eq_erase_add_single, update_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_add (a : α) (f f' : α →₀ M) : erase a (f + f') = erase a f + erase a f' := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : AddZeroClass M
    a : α
    f f' : Finsupp α M
    ⊢ Eq (Finsupp.erase a (HAdd.hAdd f f')) (HAdd.hAdd (Finsupp.erase a f) (Finsup …
  -/
  ext s; by_cases hs : s = a
    /-
      case pos
      α : Type u_1
      M : Type u_5
      inst✝ : AddZeroClass M
      a : α
      f f' : Finsupp α M
      s : α
      hs : Eq s a
      ⊢ Eq ((Finsupp.erase a (HAdd.hAdd f f')) s) ((HAdd.hAdd (Finsupp.erase a f) (F …
    -/
  · rw [hs, add_apply, erase_same, erase_same, erase_same, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    M : Type u_5
    inst✝ : AddZeroClass M
    a : α
    f f' : Finsupp α M
    s : α
    hs : Not (Eq s a)
    ⊢ Eq ((Finsupp.erase a (HAdd.hAdd f f')) s) ((HAdd.hAdd (Finsupp.erase a f) (F …
  -/
  rw [add_apply, erase_ne hs, erase_ne hs, erase_ne hs, add_apply]
  /-
    🎉 no goals
  -/


/-- `Finsupp.erase` as an `AddMonoidHom`. -/
@[simps]
def eraseAddHom (a : α) : (α →₀ M) →+ α →₀ M where
  toFun := erase a
  map_zero' := erase_zero a
  map_add' := erase_add a


@[elab_as_elim]
protected theorem induction {p : (α →₀ M) → Prop} (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), a ∉ f.support → b ≠ 0 → p f → p (single a b + f)) : p f :=
  suffices ∀ (s) (f : α →₀ M), f.support = s → p f from this _ _ rfl
  fun s =>
                                             /-
                                               α : Type u_1
                                               M : Type u_5
                                               inst✝ : AddZeroClass M
                                               p : Finsupp α M → Prop
                                               f✝ : Finsupp α M
                                               h0 : p 0
                                               ha : ∀ (a : α) (b : M) (f : Finsupp α M), Not (Membership.mem f.support a) → N …
                                               s : Finset α
                                               f : Finsupp α M
                                               hf : Eq f.support EmptyCollection.emptyCollection
                                               ⊢ p f
                                             -/
  Finset.cons_induction_on s (fun f hf => by rwa [support_eq_empty.1 hf]) fun a s has ih f hf => by
                                             /-
                                               🎉 no goals
                                             -/
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : AddZeroClass M
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), Not (Membership.mem f.support a) → N …
      s✝ : Finset α
      a : α
      s : Finset α
      has : Not (Membership.mem s a)
      ih : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hf : Eq f.support (Finset.cons a s has)
      ⊢ p f
    -/
    suffices p (single a (f a) + f.erase a) by rwa [single_add_erase] at this
    classical
      apply ha
      · rw [support_erase, mem_erase]
        exact fun H => H.1 rfl
      · rw [← mem_support_iff, hf]
        exact mem_cons_self _ _
      · apply ih _ _
        rw [support_erase, hf, Finset.erase_cons]


theorem induction₂ {p : (α →₀ M) → Prop} (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), a ∉ f.support → b ≠ 0 → p f → p (f + single a b)) : p f :=
  suffices ∀ (s) (f : α →₀ M), f.support = s → p f from this _ _ rfl
  fun s =>
                                             /-
                                               α : Type u_1
                                               M : Type u_5
                                               inst✝ : AddZeroClass M
                                               p : Finsupp α M → Prop
                                               f✝ : Finsupp α M
                                               h0 : p 0
                                               ha : ∀ (a : α) (b : M) (f : Finsupp α M), Not (Membership.mem f.support a) → N …
                                               s : Finset α
                                               f : Finsupp α M
                                               hf : Eq f.support EmptyCollection.emptyCollection
                                               ⊢ p f
                                             -/
  Finset.cons_induction_on s (fun f hf => by rwa [support_eq_empty.1 hf]) fun a s has ih f hf => by
                                             /-
                                               🎉 no goals
                                             -/
    /-
      α : Type u_1
      M : Type u_5
      inst✝ : AddZeroClass M
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), Not (Membership.mem f.support a) → N …
      s✝ : Finset α
      a : α
      s : Finset α
      has : Not (Membership.mem s a)
      ih : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hf : Eq f.support (Finset.cons a s has)
      ⊢ p f
    -/
    suffices p (f.erase a + single a (f a)) by rwa [erase_add_single] at this
    classical
      apply ha
      · rw [support_erase, mem_erase]
        exact fun H => H.1 rfl
      · rw [← mem_support_iff, hf]
        exact mem_cons_self _ _
      · apply ih _ _
        rw [support_erase, hf, Finset.erase_cons]


theorem induction_linear {p : (α →₀ M) → Prop} (f : α →₀ M) (h0 : p 0)
    (hadd : ∀ f g : α →₀ M, p f → p g → p (f + g)) (hsingle : ∀ a b, p (single a b)) : p f :=
  induction₂ f h0 fun _a _b _f _ _ w => hadd _ _ w (hsingle _ _)


/-- A finitely supported function can be built by adding up `single a b` for increasing `a`.

The theorem `induction_on_max₂` swaps the argument order in the sum. -/
theorem induction_on_max (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), (∀ c ∈ f.support, c < a) → b ≠ 0 → p f → p (single a b + f)) :
    p f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : AddZeroClass M
    inst✝ : LinearOrder α
    p : Finsupp α M → Prop
    f : Finsupp α M
    h0 : p 0
    ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
    ⊢ p f
  -/
  suffices ∀ (s) (f : α →₀ M), f.support = s → p f from this _ _ rfl
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : AddZeroClass M
    inst✝ : LinearOrder α
    p : Finsupp α M → Prop
    f : Finsupp α M
    h0 : p 0
    ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
    ⊢ ∀ (s : Finset α) (f : Finsupp α M), Eq f.support s → p f
  -/
  refine fun s => s.induction_on_max (fun f h => ?_) (fun a s hm hf f hs => ?_)
    /-
      case refine_1
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s : Finset α
      f : Finsupp α M
      h : Eq f.support EmptyCollection.emptyCollection
      ⊢ p f
    -/
  · rwa [support_eq_empty.1 h]
    /-
      🎉 no goals
    -/
  · have hs' : (erase a f).support = s := by
      rw [support_erase, hs, erase_insert (fun ha => (hm a ha).false)]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ p f
    -/
    rw [← single_add_erase a f]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ p (HAdd.hAdd (Finsupp.single a (f a)) (Finsupp.erase a f))
    -/
    refine ha _ _ _ (fun c hc => hm _ <| hs'.symm ▸ hc) ?_ (hf _ hs')
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ Ne (f a) 0
    -/
    rw [← mem_support_iff, hs]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ Membership.mem (Insert.insert a s) a
    -/
    exact mem_insert_self a s
    /-
      🎉 no goals
    -/


/-- A finitely supported function can be built by adding up `single a b` for decreasing `a`.

The theorem `induction_on_min₂` swaps the argument order in the sum. -/
theorem induction_on_min (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), (∀ c ∈ f.support, a < c) → b ≠ 0 → p f → p (single a b + f)) :
    p f :=
  induction_on_max (α := αᵒᵈ) f h0 ha


/-- A finitely supported function can be built by adding up `single a b` for increasing `a`.

The theorem `induction_on_max` swaps the argument order in the sum. -/
theorem induction_on_max₂ (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), (∀ c ∈ f.support, c < a) → b ≠ 0 → p f → p (f + single a b)) :
    p f := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : AddZeroClass M
    inst✝ : LinearOrder α
    p : Finsupp α M → Prop
    f : Finsupp α M
    h0 : p 0
    ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
    ⊢ p f
  -/
  suffices ∀ (s) (f : α →₀ M), f.support = s → p f from this _ _ rfl
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : AddZeroClass M
    inst✝ : LinearOrder α
    p : Finsupp α M → Prop
    f : Finsupp α M
    h0 : p 0
    ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
    ⊢ ∀ (s : Finset α) (f : Finsupp α M), Eq f.support s → p f
  -/
  refine fun s => s.induction_on_max (fun f h => ?_) (fun a s hm hf f hs => ?_)
    /-
      case refine_1
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s : Finset α
      f : Finsupp α M
      h : Eq f.support EmptyCollection.emptyCollection
      ⊢ p f
    -/
  · rwa [support_eq_empty.1 h]
    /-
      🎉 no goals
    -/
  · have hs' : (erase a f).support = s := by
      rw [support_erase, hs, erase_insert (fun ha => (hm a ha).false)]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ p f
    -/
    rw [← erase_add_single a f]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ p (HAdd.hAdd (Finsupp.erase a f) (Finsupp.single a (f a)))
    -/
    refine ha _ _ _ (fun c hc => hm _ <| hs'.symm ▸ hc) ?_ (hf _ hs')
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ Ne (f a) 0
    -/
    rw [← mem_support_iff, hs]
    /-
      case refine_2
      α : Type u_1
      M : Type u_5
      inst✝¹ : AddZeroClass M
      inst✝ : LinearOrder α
      p : Finsupp α M → Prop
      f✝ : Finsupp α M
      h0 : p 0
      ha : ∀ (a : α) (b : M) (f : Finsupp α M), (∀ (c : α), Membership.mem f.support …
      s✝ : Finset α
      a : α
      s : Finset α
      hm : ∀ (x : α), Membership.mem s x → LT.lt x a
      hf : ∀ (f : Finsupp α M), Eq f.support s → p f
      f : Finsupp α M
      hs : Eq f.support (Insert.insert a s)
      hs' : Eq (Finsupp.erase a f).support s
      ⊢ Membership.mem (Insert.insert a s) a
    -/
    exact mem_insert_self a s
    /-
      🎉 no goals
    -/


/-- A finitely supported function can be built by adding up `single a b` for decreasing `a`.

The theorem `induction_on_min` swaps the argument order in the sum. -/
theorem induction_on_min₂ (f : α →₀ M) (h0 : p 0)
    (ha : ∀ (a b) (f : α →₀ M), (∀ c ∈ f.support, a < c) → b ≠ 0 → p f → p (f + single a b)) :
    p f :=
  induction_on_max₂ (α := αᵒᵈ) f h0 ha


@[simp]
theorem add_closure_setOf_eq_single :
    AddSubmonoid.closure { f : α →₀ M | ∃ a b, f = single a b } = ⊤ :=
  top_unique fun x _hx =>
    Finsupp.induction x (AddSubmonoid.zero_mem _) fun a b _f _ha _hb hf =>
      AddSubmonoid.add_mem _ (AddSubmonoid.subset_closure <| ⟨a, b, rfl⟩) hf


/-- If two additive homomorphisms from `α →₀ M` are equal on each `single a b`,
then they are equal. -/
theorem addHom_ext [AddZeroClass N] ⦃f g : (α →₀ M) →+ N⦄
    (H : ∀ x y, f (single x y) = g (single x y)) : f = g := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddZeroClass M
    inst✝ : AddZeroClass N
    f g : AddMonoidHom (Finsupp α M) N
    H : ∀ (x : α) (y : M), Eq (f (Finsupp.single x y)) (g (Finsupp.single x y))
    ⊢ Eq f g
  -/
  refine AddMonoidHom.eq_of_eqOn_denseM add_closure_setOf_eq_single ?_
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddZeroClass M
    inst✝ : AddZeroClass N
    f g : AddMonoidHom (Finsupp α M) N
    H : ∀ (x : α) (y : M), Eq (f (Finsupp.single x y)) (g (Finsupp.single x y))
    ⊢ Set.EqOn (⇑f) (⇑g) (setOf fun f => Exists fun a => Exists fun b => Eq f (Fin …
  -/
  rintro _ ⟨x, y, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : AddZeroClass M
    inst✝ : AddZeroClass N
    f g : AddMonoidHom (Finsupp α M) N
    H : ∀ (x : α) (y : M), Eq (f (Finsupp.single x y)) (g (Finsupp.single x y))
    x : α
    y : M
    ⊢ Eq (f (Finsupp.single x y)) (g (Finsupp.single x y))
  -/
  apply H
  /-
    🎉 no goals
  -/


/-- If two additive homomorphisms from `α →₀ M` are equal on each `single a b`,
then they are equal.

We formulate this using equality of `AddMonoidHom`s so that `ext` tactic can apply a type-specific
extensionality lemma after this one.  E.g., if the fiber `M` is `ℕ` or `ℤ`, then it suffices to
verify `f (single a 1) = g (single a 1)`. -/
@[ext high]
theorem addHom_ext' [AddZeroClass N] ⦃f g : (α →₀ M) →+ N⦄
    (H : ∀ x, f.comp (singleAddHom x) = g.comp (singleAddHom x)) : f = g :=
  addHom_ext fun x => DFunLike.congr_fun (H x)


theorem mulHom_ext [MulOneClass N] ⦃f g : Multiplicative (α →₀ M) →* N⦄
    (H : ∀ x y, f (Multiplicative.ofAdd <| single x y) = g (Multiplicative.ofAdd <| single x y)) :
    f = g :=
  MonoidHom.ext <|
    DFunLike.congr_fun <| by
      have := @addHom_ext α M (Additive N) _ _
        (MonoidHom.toAdditive'' f) (MonoidHom.toAdditive'' g) H
      /-
        α : Type u_1
        M : Type u_5
        N : Type u_7
        inst✝¹ : AddZeroClass M
        inst✝ : MulOneClass N
        f g : MonoidHom (Multiplicative (Finsupp α M)) N
        H : ∀ (x : α) (y : M), Eq (f (Multiplicative.ofAdd (Finsupp.single x y))) (g ( …
        this : Eq (MonoidHom.toAdditive'' f) (MonoidHom.toAdditive'' g)
        ⊢ Eq f g
      -/
      ext
      /-
        case h
        α : Type u_1
        M : Type u_5
        N : Type u_7
        inst✝¹ : AddZeroClass M
        inst✝ : MulOneClass N
        f g : MonoidHom (Multiplicative (Finsupp α M)) N
        H : ∀ (x : α) (y : M), Eq (f (Multiplicative.ofAdd (Finsupp.single x y))) (g ( …
        this : Eq (MonoidHom.toAdditive'' f) (MonoidHom.toAdditive'' g)
        x✝ : Multiplicative (Finsupp α M)
        ⊢ Eq (f x✝) (g x✝)
      -/
      rw [DFunLike.ext_iff] at this
      /-
        case h
        α : Type u_1
        M : Type u_5
        N : Type u_7
        inst✝¹ : AddZeroClass M
        inst✝ : MulOneClass N
        f g : MonoidHom (Multiplicative (Finsupp α M)) N
        H : ∀ (x : α) (y : M), Eq (f (Multiplicative.ofAdd (Finsupp.single x y))) (g ( …
        this : ∀ (x : Finsupp α M), Eq ((MonoidHom.toAdditive'' f) x) ((MonoidHom.toAd …
        x✝ : Multiplicative (Finsupp α M)
        ⊢ Eq (f x✝) (g x✝)
      -/
      apply this
      /-
        🎉 no goals
      -/


@[ext]
theorem mulHom_ext' [MulOneClass N] {f g : Multiplicative (α →₀ M) →* N}
    (H : ∀ x, f.comp (AddMonoidHom.toMultiplicative (singleAddHom x)) =
              g.comp (AddMonoidHom.toMultiplicative (singleAddHom x))) :
    f = g :=
  mulHom_ext fun x => DFunLike.congr_fun (H x)


theorem mapRange_add [AddZeroClass N] {f : M → N} {hf : f 0 = 0}
    (hf' : ∀ x y, f (x + y) = f x + f y) (v₁ v₂ : α →₀ M) :
    mapRange f hf (v₁ + v₂) = mapRange f hf v₁ + mapRange f hf v₂ :=
                  /-
                    α : Type u_1
                    M : Type u_5
                    N : Type u_7
                    inst✝¹ : AddZeroClass M
                    inst✝ : AddZeroClass N
                    f : M → N
                    hf : Eq (f 0) 0
                    hf' : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                    v₁ v₂ : Finsupp α M
                    x✝ : α
                    ⊢ Eq ((Finsupp.mapRange f hf (HAdd.hAdd v₁ v₂)) x✝) ((HAdd.hAdd (Finsupp.mapRa …
                  -/
  ext fun _ => by simp only [hf', add_apply, mapRange_apply]
                  /-
                    🎉 no goals
                  -/


theorem mapRange_add' [AddZeroClass N] [FunLike β M N] [AddMonoidHomClass β M N]
    {f : β} (v₁ v₂ : α →₀ M) :
    mapRange f (map_zero f) (v₁ + v₂) = mapRange f (map_zero f) v₁ + mapRange f (map_zero f) v₂ :=
  mapRange_add (map_add f) v₁ v₂


/-- Bundle `Finsupp.embDomain f` as an additive map from `α →₀ M` to `β →₀ M`. -/
@[simps]
def embDomain.addMonoidHom (f : α ↪ β) : (α →₀ M) →+ β →₀ M where
  toFun v := embDomain f v
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    ι : Type u_4
                    M : Type u_5
                    M' : Type u_6
                    N : Type u_7
                    P : Type u_8
                    G : Type u_9
                    H : Type u_10
                    R : Type u_11
                    S : Type u_12
                    inst✝ : AddZeroClass M
                    f : Function.Embedding α β
                    ⊢ Eq ((fun v => Finsupp.embDomain f v) 0) 0
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
  map_add' v w := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddZeroClass M
      f : Function.Embedding α β
      v w : Finsupp α M
      ⊢ Eq ({ toFun := fun v => Finsupp.embDomain f v, map_zero' := ⋯ }.toFun (HAdd. …
    -/
    ext b
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝ : AddZeroClass M
      f : Function.Embedding α β
      v w : Finsupp α M
      b : β
      ⊢ Eq (({ toFun := fun v => Finsupp.embDomain f v, map_zero' := ⋯ }.toFun (HAdd …
    -/
    by_cases h : b ∈ Set.range f
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝ : AddZeroClass M
        f : Function.Embedding α β
        v w : Finsupp α M
        b : β
        h : Membership.mem (Set.range ⇑f) b
        ⊢ Eq (({ toFun := fun v => Finsupp.embDomain f v, map_zero' := ⋯ }.toFun (HAdd …
      -/
    · rcases h with ⟨a, rfl⟩
      /-
        case pos.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        ι : Type u_4
        M : Type u_5
        M' : Type u_6
        N : Type u_7
        P : Type u_8
        G : Type u_9
        H : Type u_10
        R : Type u_11
        S : Type u_12
        inst✝ : AddZeroClass M
        f : Function.Embedding α β
        v w : Finsupp α M
        a : α
        ⊢ Eq (({ toFun := fun v => Finsupp.embDomain f v, map_zero' := ⋯ }.toFun (HAdd …
      -/
      simp
      /-
        🎉 no goals
      -/
    · simp only [Set.mem_range, not_exists, coe_add, Pi.add_apply,
        embDomain_notin_range _ _ _ h, add_zero]


@[simp]
theorem embDomain_add (f : α ↪ β) (v w : α →₀ M) :
    embDomain f (v + w) = embDomain f v + embDomain f w :=
  (embDomain.addMonoidHom f).map_add v w


/-- Note the general `SMul` instance for `Finsupp` doesn't apply as `ℕ` is not distributive
unless `β i`'s addition is commutative. -/
instance instNatSMul : SMul ℕ (α →₀ M) :=
  ⟨fun n v => v.mapRange (n • ·) (nsmul_zero _)⟩


instance instAddMonoid : AddMonoid (α →₀ M) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ => rfl


instance instAddCommMonoid [AddCommMonoid M] : AddCommMonoid (α →₀ M) :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { DFunLike.coe_injective.addCommMonoid DFunLike.coe coe_zero coe_add (fun _ _ => rfl) with
    toAddMonoid := Finsupp.instAddMonoid }


instance instNeg [NegZeroClass G] : Neg (α →₀ G) :=
  ⟨mapRange Neg.neg neg_zero⟩


@[simp, norm_cast] lemma coe_neg [NegZeroClass G] (g : α →₀ G) : ⇑(-g) = -g := rfl


theorem neg_apply [NegZeroClass G] (g : α →₀ G) (a : α) : (-g) a = -g a :=
  rfl


theorem mapRange_neg [NegZeroClass G] [NegZeroClass H] {f : G → H} {hf : f 0 = 0}
    (hf' : ∀ x, f (-x) = -f x) (v : α →₀ G) : mapRange f hf (-v) = -mapRange f hf v :=
                  /-
                    α : Type u_1
                    G : Type u_9
                    H : Type u_10
                    inst✝¹ : NegZeroClass G
                    inst✝ : NegZeroClass H
                    f : G → H
                    hf : Eq (f 0) 0
                    hf' : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
                    v : Finsupp α G
                    x✝ : α
                    ⊢ Eq ((Finsupp.mapRange f hf (Neg.neg v)) x✝) ((Neg.neg (Finsupp.mapRange f hf …
                  -/
  ext fun _ => by simp only [hf', neg_apply, mapRange_apply]
                  /-
                    🎉 no goals
                  -/


theorem mapRange_neg' [AddGroup G] [SubtractionMonoid H] [FunLike β G H] [AddMonoidHomClass β G H]
    {f : β} (v : α →₀ G) :
    mapRange f (map_zero f) (-v) = -mapRange f (map_zero f) v :=
  mapRange_neg (map_neg f) v


instance instSub [SubNegZeroMonoid G] : Sub (α →₀ G) :=
  ⟨zipWith Sub.sub (sub_zero _)⟩


@[simp, norm_cast] lemma coe_sub [SubNegZeroMonoid G] (g₁ g₂ : α →₀ G) : ⇑(g₁ - g₂) = g₁ - g₂ := rfl


theorem sub_apply [SubNegZeroMonoid G] (g₁ g₂ : α →₀ G) (a : α) : (g₁ - g₂) a = g₁ a - g₂ a :=
  rfl


theorem mapRange_sub [SubNegZeroMonoid G] [SubNegZeroMonoid H] {f : G → H} {hf : f 0 = 0}
    (hf' : ∀ x y, f (x - y) = f x - f y) (v₁ v₂ : α →₀ G) :
    mapRange f hf (v₁ - v₂) = mapRange f hf v₁ - mapRange f hf v₂ :=
                  /-
                    α : Type u_1
                    G : Type u_9
                    H : Type u_10
                    inst✝¹ : SubNegZeroMonoid G
                    inst✝ : SubNegZeroMonoid H
                    f : G → H
                    hf : Eq (f 0) 0
                    hf' : ∀ (x y : G), Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
                    v₁ v₂ : Finsupp α G
                    x✝ : α
                    ⊢ Eq ((Finsupp.mapRange f hf (HSub.hSub v₁ v₂)) x✝) ((HSub.hSub (Finsupp.mapRa …
                  -/
  ext fun _ => by simp only [hf', sub_apply, mapRange_apply]
                  /-
                    🎉 no goals
                  -/


theorem mapRange_sub' [AddGroup G] [SubtractionMonoid H] [FunLike β G H] [AddMonoidHomClass β G H]
    {f : β} (v₁ v₂ : α →₀ G) :
    mapRange f (map_zero f) (v₁ - v₂) = mapRange f (map_zero f) v₁ - mapRange f (map_zero f) v₂ :=
  mapRange_sub (map_sub f) v₁ v₂


/-- Note the general `SMul` instance for `Finsupp` doesn't apply as `ℤ` is not distributive
unless `β i`'s addition is commutative. -/
instance instIntSMul [AddGroup G] : SMul ℤ (α →₀ G) :=
  ⟨fun n v => v.mapRange (n • ·) (zsmul_zero _)⟩


instance instAddGroup [AddGroup G] : AddGroup (α →₀ G) :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { DFunLike.coe_injective.addGroup DFunLike.coe coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl)
      fun _ _ => rfl with
    toAddMonoid := Finsupp.instAddMonoid }


instance instAddCommGroup [AddCommGroup G] : AddCommGroup (α →₀ G) :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { DFunLike.coe_injective.addCommGroup DFunLike.coe coe_zero coe_add coe_neg coe_sub
      (fun _ _ => rfl) fun _ _ => rfl with
    toAddGroup := Finsupp.instAddGroup }


theorem single_add_single_eq_single_add_single [AddCommMonoid M] {k l m n : α} {u v : M}
    (hu : u ≠ 0) (hv : v ≠ 0) :
    single k u + single l v = single m u + single n v ↔
      (k = m ∧ l = n) ∨ (u = v ∧ k = n ∧ l = m) ∨ (u + v = 0 ∧ k = l ∧ m = n) := by
  classical
    simp_rw [DFunLike.ext_iff, coe_add, single_eq_pi_single, ← funext_iff]
    exact Pi.single_add_single_eq_single_add_single hu hv


@[simp]
theorem support_neg [AddGroup G] (f : α →₀ G) : support (-f) = support f :=
  Finset.Subset.antisymm support_mapRange
    (calc
      support f = support (- -f) := congr_arg support (neg_neg _).symm
      _ ⊆ support (-f) := support_mapRange
      )


theorem support_sub [DecidableEq α] [AddGroup G] {f g : α →₀ G} :
    support (f - g) ⊆ support f ∪ support g := by
  /-
    α : Type u_1
    G : Type u_9
    inst✝¹ : DecidableEq α
    inst✝ : AddGroup G
    f g : Finsupp α G
    ⊢ HasSubset.Subset (HSub.hSub f g).support (Union.union f.support g.support)
  -/
  rw [sub_eq_add_neg, ← support_neg g]
  /-
    α : Type u_1
    G : Type u_9
    inst✝¹ : DecidableEq α
    inst✝ : AddGroup G
    f g : Finsupp α G
    ⊢ HasSubset.Subset (HAdd.hAdd f (Neg.neg g)).support (Union.union f.support (N …
  -/
  exact support_add
  /-
    🎉 no goals
  -/


theorem erase_eq_sub_single [AddGroup G] (f : α →₀ G) (a : α) : f.erase a = f - single a (f a) := by
  /-
    α : Type u_1
    G : Type u_9
    inst✝ : AddGroup G
    f : Finsupp α G
    a : α
    ⊢ Eq (Finsupp.erase a f) (HSub.hSub f (Finsupp.single a (f a)))
  -/
  ext a'
  /-
    case h
    α : Type u_1
    G : Type u_9
    inst✝ : AddGroup G
    f : Finsupp α G
    a a' : α
    ⊢ Eq ((Finsupp.erase a f) a') ((HSub.hSub f (Finsupp.single a (f a))) a')
  -/
  rcases eq_or_ne a a' with (rfl | h)
    /-
      case h.inl
      α : Type u_1
      G : Type u_9
      inst✝ : AddGroup G
      f : Finsupp α G
      a : α
      ⊢ Eq ((Finsupp.erase a f) a) ((HSub.hSub f (Finsupp.single a (f a))) a)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      G : Type u_9
      inst✝ : AddGroup G
      f : Finsupp α G
      a a' : α
      h : Ne a a'
      ⊢ Eq ((Finsupp.erase a f) a') ((HSub.hSub f (Finsupp.single a (f a))) a')
    -/
  · simp [erase_ne h.symm, single_eq_of_ne h]
    /-
      🎉 no goals
    -/


theorem update_eq_sub_add_single [AddGroup G] (f : α →₀ G) (a : α) (b : G) :
    f.update a b = f - single a (f a) + single a b := by
  /-
    α : Type u_1
    G : Type u_9
    inst✝ : AddGroup G
    f : Finsupp α G
    a : α
    b : G
    ⊢ Eq (f.update a b) (HAdd.hAdd (HSub.hSub f (Finsupp.single a (f a))) (Finsupp …
  -/
  rw [update_eq_erase_add_single, erase_eq_sub_single]
  /-
    🎉 no goals
  -/



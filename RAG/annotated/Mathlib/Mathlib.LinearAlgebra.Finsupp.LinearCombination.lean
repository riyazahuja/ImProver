/-- Interprets (l : α →₀ R) as a linear combination of the elements in the family (v : α → M) and
    evaluates this linear combination. -/
def linearCombination : (α →₀ R) →ₗ[R] M :=
  Finsupp.lsum ℕ fun i => LinearMap.id.smulRight (v i)


@[deprecated (since := "2024-08-29")] noncomputable alias total := linearCombination


theorem linearCombination_apply (l : α →₀ R) : linearCombination R v l = l.sum fun i a => a • v i :=
  rfl


@[deprecated (since := "2024-08-29")] alias total_apply := linearCombination_apply


theorem linearCombination_apply_of_mem_supported {l : α →₀ R} {s : Finset α}
    (hs : l ∈ supported R R (↑s : Set α)) : linearCombination R v l = s.sum fun i => l i • v i :=
  Finset.sum_subset hs fun x _ hxg =>
                          /-
                            α : Type u_1
                            M : Type u_2
                            R : Type u_5
                            inst✝² : Semiring R
                            inst✝¹ : AddCommMonoid M
                            inst✝ : Module R M
                            v : α → M
                            l : Finsupp α R
                            s : Finset α
                            hs : Membership.mem (Finsupp.supported R R ↑s) l
                            x : α
                            x✝ : Membership.mem s x
                            hxg : Not (Membership.mem l.support x)
                            ⊢ Eq (HSMul.hSMul (l x) (v x)) 0
                          -/
    show l x • v x = 0 by rw [not_mem_support_iff.1 hxg, zero_smul]
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-08-29")] alias total_apply_of_mem_supported :=
  linearCombination_apply_of_mem_supported


@[simp]
theorem linearCombination_single (c : R) (a : α) :
    linearCombination R v (single a c) = c • v a := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    c : R
    a : α
    ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.single a c)) (HSMul.hSMul c (v  …
  -/
  simp [linearCombination_apply, sum_single_index]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_single := linearCombination_single


theorem linearCombination_zero_apply (x : α →₀ R) : (linearCombination R (0 : α → M)) x = 0 := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : Finsupp α R
    ⊢ Eq ((Finsupp.linearCombination R 0) x) 0
  -/
  simp [linearCombination_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_zero_apply := linearCombination_zero_apply


@[simp]
theorem linearCombination_zero : linearCombination R (0 : α → M) = 0 :=
  LinearMap.ext (linearCombination_zero_apply R)


@[deprecated (since := "2024-08-29")] alias total_zero := linearCombination_zero


theorem linearCombination_linear_comp (f : M →ₗ[R] M') :
    linearCombination R (f ∘ v) = f ∘ₗ linearCombination R v := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v : α → M
    f : LinearMap (RingHom.id R) M M'
    ⊢ Eq (Finsupp.linearCombination R (Function.comp (⇑f) v)) (f.comp (Finsupp.lin …
  -/
  ext
  /-
    case h.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v : α → M
    f : LinearMap (RingHom.id R) M M'
    a✝ : α
    ⊢ Eq (((Finsupp.linearCombination R (Function.comp (⇑f) v)).comp (Finsupp.lsin …
  -/
  simp [linearCombination_apply]
  /-
    🎉 no goals
  -/


theorem apply_linearCombination (f : M →ₗ[R] M') (v) (l : α →₀ R) :
    f (linearCombination R v l) = linearCombination R (f ∘ v) l :=
  congr($(linearCombination_linear_comp R f) l).symm


@[deprecated (since := "2024-08-29")] alias apply_total := apply_linearCombination


theorem apply_linearCombination_id (f : M →ₗ[R] M') (l : M →₀ R) :
    f (linearCombination R _root_.id l) = linearCombination R f l :=
  apply_linearCombination ..


@[deprecated (since := "2024-08-29")] alias apply_total_id := apply_linearCombination_id


theorem linearCombination_unique [Unique α] (l : α →₀ R) (v : α → M) :
    linearCombination R v l = l default • v default := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Unique α
    l : Finsupp α R
    v : α → M
    ⊢ Eq ((Finsupp.linearCombination R v) l) (HSMul.hSMul (l Inhabited.default) (v …
  -/
  rw [← linearCombination_single, ← unique_single l]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_unique := linearCombination_unique


theorem linearCombination_surjective (h : Function.Surjective v) :
    Function.Surjective (linearCombination R v) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    h : Function.Surjective v
    ⊢ Function.Surjective ⇑(Finsupp.linearCombination R v)
  -/
  intro x
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    h : Function.Surjective v
    x : M
    ⊢ Exists fun a => Eq ((Finsupp.linearCombination R v) a) x
  -/
  obtain ⟨y, hy⟩ := h x
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    h : Function.Surjective v
    x : M
    y : α
    hy : Eq (v y) x
    ⊢ Exists fun a => Eq ((Finsupp.linearCombination R v) a) x
  -/
  exact ⟨Finsupp.single y 1, by simp [hy]⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_surjective := linearCombination_surjective


theorem linearCombination_range (h : Function.Surjective v) :
    LinearMap.range (linearCombination R v) = ⊤ :=
  range_eq_top.2 <| linearCombination_surjective R h


@[deprecated (since := "2024-08-29")] alias total_range := linearCombination_range


/-- Any module is a quotient of a free module. This is stated as surjectivity of
`Finsupp.linearCombination R id : (M →₀ R) →ₗ[R] M`. -/
theorem linearCombination_id_surjective (M) [AddCommMonoid M] [Module R M] :
    Function.Surjective (linearCombination R (id : M → M)) :=
  linearCombination_surjective R Function.surjective_id


@[deprecated (since := "2024-08-29")] alias total_id_surjective := linearCombination_id_surjective


theorem range_linearCombination : LinearMap.range (linearCombination R v) = span R (range v) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    ⊢ Eq (LinearMap.range (Finsupp.linearCombination R v)) (Submodule.span R (Set. …
  -/
  ext x
  /-
    case h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    x : M
    ⊢ Iff (Membership.mem (LinearMap.range (Finsupp.linearCombination R v)) x) (Me …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      ⊢ Membership.mem (LinearMap.range (Finsupp.linearCombination R v)) x → Members …
    -/
  · intro hx
    /-
      case h.mp
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      hx : Membership.mem (LinearMap.range (Finsupp.linearCombination R v)) x
      ⊢ Membership.mem (Submodule.span R (Set.range v)) x
    -/
    rw [LinearMap.mem_range] at hx
    /-
      case h.mp
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      hx : Exists fun y => Eq ((Finsupp.linearCombination R v) y) x
      ⊢ Membership.mem (Submodule.span R (Set.range v)) x
    -/
    rcases hx with ⟨l, hl⟩
    /-
      case h.mp.intro
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      l : Finsupp α R
      hl : Eq ((Finsupp.linearCombination R v) l) x
      ⊢ Membership.mem (Submodule.span R (Set.range v)) x
    -/
    rw [← hl]
    /-
      case h.mp.intro
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      l : Finsupp α R
      hl : Eq ((Finsupp.linearCombination R v) l) x
      ⊢ Membership.mem (Submodule.span R (Set.range v)) ((Finsupp.linearCombination  …
    -/
    rw [linearCombination_apply]
    /-
      case h.mp.intro
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      l : Finsupp α R
      hl : Eq ((Finsupp.linearCombination R v) l) x
      ⊢ Membership.mem (Submodule.span R (Set.range v)) (l.sum fun i a => HSMul.hSMu …
    -/
    exact sum_mem fun i _ => Submodule.smul_mem _ _ (subset_span (mem_range_self i))
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      ⊢ Membership.mem (Submodule.span R (Set.range v)) x → Membership.mem (LinearMa …
    -/
  · apply span_le.2
    /-
      case h.mpr.a
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x : M
      ⊢ HasSubset.Subset (Set.range v) ↑(LinearMap.range (Finsupp.linearCombination  …
    -/
    intro x hx
    /-
      case h.mpr.a
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x✝ x : M
      hx : Membership.mem (Set.range v) x
      ⊢ Membership.mem (↑(LinearMap.range (Finsupp.linearCombination R v))) x
    -/
    rcases hx with ⟨i, hi⟩
    /-
      case h.mpr.a.intro
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x✝ x : M
      i : α
      hi : Eq (v i) x
      ⊢ Membership.mem (↑(LinearMap.range (Finsupp.linearCombination R v))) x
    -/
    rw [SetLike.mem_coe, LinearMap.mem_range]
    /-
      case h.mpr.a.intro
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x✝ x : M
      i : α
      hi : Eq (v i) x
      ⊢ Exists fun y => Eq ((Finsupp.linearCombination R v) y) x
    -/
    use Finsupp.single i 1
    /-
      case h
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      x✝ x : M
      i : α
      hi : Eq (v i) x
      ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.single i 1)) x
    -/
    simp [hi]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias range_total := range_linearCombination


theorem lmapDomain_linearCombination (f : α → α') (g : M →ₗ[R] M') (h : ∀ i, g (v i) = v' (f i)) :
    (linearCombination R v').comp (lmapDomain R R f) = g.comp (linearCombination R v) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v : α → M
    v' : α' → M'
    f : α → α'
    g : LinearMap (RingHom.id R) M M'
    h : ∀ (i : α), Eq (g (v i)) (v' (f i))
    ⊢ Eq ((Finsupp.linearCombination R v').comp (Finsupp.lmapDomain R R f)) (g.com …
  -/
  ext l
  /-
    case h.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v : α → M
    v' : α' → M'
    f : α → α'
    g : LinearMap (RingHom.id R) M M'
    h : ∀ (i : α), Eq (g (v i)) (v' (f i))
    l : α
    ⊢ Eq ((((Finsupp.linearCombination R v').comp (Finsupp.lmapDomain R R f)).comp …
  -/
  simp [linearCombination_apply, Finsupp.sum_mapDomain_index, add_smul, h]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias lmapDomain_total := lmapDomain_linearCombination


theorem linearCombination_comp_lmapDomain (f : α → α') :
    (linearCombination R v').comp (Finsupp.lmapDomain R R f) = linearCombination R (v' ∘ f) := by
  /-
    α : Type u_1
    R : Type u_5
    inst✝² : Semiring R
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v' : α' → M'
    f : α → α'
    ⊢ Eq ((Finsupp.linearCombination R v').comp (Finsupp.lmapDomain R R f)) (Finsu …
  -/
  ext
  /-
    case h.h
    α : Type u_1
    R : Type u_5
    inst✝² : Semiring R
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v' : α' → M'
    f : α → α'
    a✝ : α
    ⊢ Eq ((((Finsupp.linearCombination R v').comp (Finsupp.lmapDomain R R f)).comp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_comp_lmapDomain :=
  linearCombination_comp_lmapDomain


@[simp]
theorem linearCombination_embDomain (f : α ↪ α') (l : α →₀ R) :
    (linearCombination R v') (embDomain f l) = (linearCombination R (v' ∘ f)) l := by
  /-
    α : Type u_1
    R : Type u_5
    inst✝² : Semiring R
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v' : α' → M'
    f : Function.Embedding α α'
    l : Finsupp α R
    ⊢ Eq ((Finsupp.linearCombination R v') (Finsupp.embDomain f l)) ((Finsupp.line …
  -/
  simp [linearCombination_apply, Finsupp.sum, support_embDomain, embDomain_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_embDomain := linearCombination_embDomain


@[simp]
theorem linearCombination_mapDomain (f : α → α') (l : α →₀ R) :
    (linearCombination R v') (mapDomain f l) = (linearCombination R (v' ∘ f)) l :=
  LinearMap.congr_fun (linearCombination_comp_lmapDomain _ _) l


@[deprecated (since := "2024-08-29")] alias total_mapDomain := linearCombination_mapDomain


@[simp]
theorem linearCombination_equivMapDomain (f : α ≃ α') (l : α →₀ R) :
    (linearCombination R v') (equivMapDomain f l) = (linearCombination R (v' ∘ f)) l := by
  /-
    α : Type u_1
    R : Type u_5
    inst✝² : Semiring R
    α' : Type u_7
    M' : Type u_8
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    v' : α' → M'
    f : Equiv α α'
    l : Finsupp α R
    ⊢ Eq ((Finsupp.linearCombination R v') (Finsupp.equivMapDomain f l)) ((Finsupp …
  -/
  rw [equivMapDomain_eq_mapDomain, linearCombination_mapDomain]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_equivMapDomain := linearCombination_equivMapDomain


/-- A version of `Finsupp.range_linearCombination` which is useful for going in the other
direction -/
theorem span_eq_range_linearCombination (s : Set M) :
    span R s = LinearMap.range (linearCombination R ((↑) : s → M)) := by
  /-
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    ⊢ Eq (Submodule.span R s) (LinearMap.range (Finsupp.linearCombination R Subtyp …
  -/
  rw [range_linearCombination, Subtype.range_coe_subtype, Set.setOf_mem_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias span_eq_range_total := span_eq_range_linearCombination


theorem mem_span_iff_linearCombination (s : Set M) (x : M) :
    x ∈ span R s ↔ ∃ l : s →₀ R, linearCombination R (↑) l = x :=
  (SetLike.ext_iff.1 <| span_eq_range_linearCombination _ _) x


@[deprecated (since := "2024-08-29")] alias mem_span_iff_total := mem_span_iff_linearCombination


theorem mem_span_range_iff_exists_finsupp {v : α → M} {x : M} :
    x ∈ span R (range v) ↔ ∃ c : α →₀ R, (c.sum fun i a => a • v i) = x := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    x : M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.range v)) x) (Exists fun c => Eq  …
  -/
  simp only [← Finsupp.range_linearCombination, LinearMap.mem_range, linearCombination_apply]
  /-
    🎉 no goals
  -/


theorem span_image_eq_map_linearCombination (s : Set α) :
    span R (v '' s) = Submodule.map (linearCombination R v) (supported R R s) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    s : Set α
    ⊢ Eq (Submodule.span R (Set.image v s)) (Submodule.map (Finsupp.linearCombinat …
  -/
  apply span_eq_of_le
    /-
      case h₁
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      ⊢ HasSubset.Subset (Set.image v s) ↑(Submodule.map (Finsupp.linearCombination  …
    -/
  · intro x hx
    /-
      case h₁
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      x : M
      hx : Membership.mem (Set.image v s) x
      ⊢ Membership.mem (↑(Submodule.map (Finsupp.linearCombination R v) (Finsupp.sup …
    -/
    rw [Set.mem_image] at hx
    /-
      case h₁
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      x : M
      hx : Exists fun x_1 => And (Membership.mem s x_1) (Eq (v x_1) x)
      ⊢ Membership.mem (↑(Submodule.map (Finsupp.linearCombination R v) (Finsupp.sup …
    -/
    apply Exists.elim hx
    /-
      case h₁
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      x : M
      hx : Exists fun x_1 => And (Membership.mem s x_1) (Eq (v x_1) x)
      ⊢ ∀ (a : α), And (Membership.mem s a) (Eq (v a) x) → Membership.mem (↑(Submodu …
    -/
    intro i hi
    /-
      case h₁
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      x : M
      hx : Exists fun x_1 => And (Membership.mem s x_1) (Eq (v x_1) x)
      i : α
      hi : And (Membership.mem s i) (Eq (v i) x)
      ⊢ Membership.mem (↑(Submodule.map (Finsupp.linearCombination R v) (Finsupp.sup …
    -/
    exact ⟨_, Finsupp.single_mem_supported R 1 hi.1, by simp [hi.2]⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      ⊢ LE.le (Submodule.map (Finsupp.linearCombination R v) (Finsupp.supported R R  …
    -/
  · refine map_le_iff_le_comap.2 fun z hz => ?_
    have : ∀ i, z i • v i ∈ span R (v '' s) := by
      intro c
      haveI := Classical.decPred fun x => x ∈ s
      by_cases h : c ∈ s
      · exact smul_mem _ _ (subset_span (Set.mem_image_of_mem _ h))
      · simp [(Finsupp.mem_supported' R _).1 hz _ h]
    -- Porting note: `rw` is required to infer metavariables in `sum_mem`.
    /-
      case h₂
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      z : Finsupp α R
      hz : Membership.mem (Finsupp.supported R R s) z
      this : ∀ (i : α), Membership.mem (Submodule.span R (Set.image v s)) (HSMul.hSM …
      ⊢ Membership.mem (Submodule.comap (Finsupp.linearCombination R v) (Submodule.s …
    -/
    rw [mem_comap, linearCombination_apply]
    /-
      case h₂
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      z : Finsupp α R
      hz : Membership.mem (Finsupp.supported R R s) z
      this : ∀ (i : α), Membership.mem (Submodule.span R (Set.image v s)) (HSMul.hSM …
      ⊢ Membership.mem (Submodule.span R (Set.image v s)) (z.sum fun i a => HSMul.hS …
    -/
    refine sum_mem ?_
    /-
      case h₂
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      v : α → M
      s : Set α
      z : Finsupp α R
      hz : Membership.mem (Finsupp.supported R R s) z
      this : ∀ (i : α), Membership.mem (Submodule.span R (Set.image v s)) (HSMul.hSM …
      ⊢ ∀ (c : α), Membership.mem z.support c → Membership.mem (Submodule.span R (Se …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias span_image_eq_map_total :=
  span_image_eq_map_linearCombination


theorem mem_span_image_iff_linearCombination {s : Set α} {x : M} :
    x ∈ span R (v '' s) ↔ ∃ l ∈ supported R R s, linearCombination R v l = x := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    s : Set α
    x : M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.image v s)) x) (Exists fun l => A …
  -/
  rw [span_image_eq_map_linearCombination]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    s : Set α
    x : M
    ⊢ Iff (Membership.mem (Submodule.map (Finsupp.linearCombination R v) (Finsupp. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias mem_span_image_iff_total :=
  mem_span_image_iff_linearCombination


theorem linearCombination_option (v : Option α → M) (f : Option α →₀ R) :
    linearCombination R v f =
      f none • v none + linearCombination R (v ∘ Option.some) f.some := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : Option α → M
    f : Finsupp (Option α) R
    ⊢ Eq ((Finsupp.linearCombination R v) f) (HAdd.hAdd (HSMul.hSMul (f Option.non …
  -/
  rw [linearCombination_apply, sum_option_index_smul, linearCombination_apply]; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[deprecated (since := "2024-08-29")] alias total_option := linearCombination_option


theorem linearCombination_linearCombination {α β : Type*} (A : α → M) (B : β → α →₀ R)
    (f : β →₀ R) : linearCombination R A (linearCombination R B f) =
      linearCombination R (fun b => linearCombination R A (B b)) f := by
  classical
  simp only [linearCombination_apply]
  apply induction_linear f
  · simp only [sum_zero_index]
  · intro f₁ f₂ h₁ h₂
    simp [sum_add_index, h₁, h₂, add_smul]
  · simp [sum_single_index, sum_smul_index, smul_sum, mul_smul]


@[deprecated (since := "2024-08-29")] alias total_total := linearCombination_linearCombination


@[simp]
theorem linearCombination_fin_zero (f : Fin 0 → M) : linearCombination R f = 0 := by
  /-
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Fin 0 → M
    ⊢ Eq (Finsupp.linearCombination R f) 0
  -/
  ext i
  /-
    case h.h
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Fin 0 → M
    i : Fin 0
    ⊢ Eq (((Finsupp.linearCombination R f).comp (Finsupp.lsingle i)) 1) ((LinearMa …
  -/
  apply finZeroElim i
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_fin_zero := linearCombination_fin_zero


/-- `Finsupp.linearCombinationOn M v s` interprets `p : α →₀ R` as a linear combination of a
subset of the vectors in `v`, mapping it to the span of those vectors.

The subset is indicated by a set `s : Set α` of indices.
-/
def linearCombinationOn (s : Set α) : supported R R s →ₗ[R] span R (v '' s) :=
  LinearMap.codRestrict _ ((linearCombination _ v).comp (Submodule.subtype (supported R R s)))
    fun ⟨l, hl⟩ => (mem_span_image_iff_linearCombination _).2 ⟨l, hl, rfl⟩


@[deprecated (since := "2024-08-29")] noncomputable alias totalOn := linearCombinationOn


theorem linearCombinationOn_range (s : Set α) :
    LinearMap.range (linearCombinationOn α M R v s) = ⊤ := by
  rw [linearCombinationOn, LinearMap.range_eq_map, LinearMap.map_codRestrict,
    ← LinearMap.range_le_iff_comap, range_subtype, Submodule.map_top, LinearMap.range_comp,
    range_subtype]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    s : Set α
    ⊢ LE.le (Submodule.span R (Set.image v s)) (Submodule.map (Finsupp.linearCombi …
  -/
  exact (span_image_eq_map_linearCombination _ _).le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias totalOn_range := linearCombinationOn_range


theorem linearCombination_comp (f : α' → α) :
    linearCombination R (v ∘ f) = (linearCombination R v).comp (lmapDomain R R f) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    v : α → M
    f : α' → α
    ⊢ Eq (Finsupp.linearCombination R (Function.comp v f)) ((Finsupp.linearCombina …
  -/
  ext
  /-
    case h.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    v : α → M
    f : α' → α
    a✝ : α'
    ⊢ Eq (((Finsupp.linearCombination R (Function.comp v f)).comp (Finsupp.lsingle …
  -/
  simp [linearCombination_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_comp := linearCombination_comp


theorem linearCombination_comapDomain (f : α → α') (l : α' →₀ R)
    (hf : Set.InjOn f (f ⁻¹' ↑l.support)) : linearCombination R v (Finsupp.comapDomain f l hf) =
      (l.support.preimage f hf).sum fun i => l (f i) • v i := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    v : α → M
    f : α → α'
    l : Finsupp α' R
    hf : Set.InjOn f (Set.preimage f ↑l.support)
    ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.comapDomain f l hf)) ((l.suppor …
  -/
  rw [linearCombination_apply]; rfl
                                /-
                                  🎉 no goals
                                -/


@[deprecated (since := "2024-08-29")] alias total_comapDomain := linearCombination_comapDomain


theorem linearCombination_onFinset {s : Finset α} {f : α → R} (g : α → M)
    (hf : ∀ a, f a ≠ 0 → a ∈ s) :
    linearCombination R g (Finsupp.onFinset s f hf) = Finset.sum s fun x : α => f x • g x := by
  classical
  simp only [linearCombination_apply, Finsupp.sum, Finsupp.onFinset_apply, Finsupp.support_onFinset]
  rw [Finset.sum_filter_of_ne]
  intro x _ h
  contrapose! h
  simp [h]


@[deprecated (since := "2024-08-29")] alias total_onFinset := linearCombination_onFinset


/-- `Fintype.linearCombination R S v f` is the linear combination of vectors in `v` with weights
in `f`. This variant of `Finsupp.linearCombination` is defined on fintype indexed vectors.

This map is linear in `v` if `R` is commutative, and always linear in `f`.
See note [bundled maps over different rings] for why separate `R` and `S` semirings are used.
-/
protected def Fintype.linearCombination : (α → M) →ₗ[S] (α → R) →ₗ[R] M where
  toFun v :=
    { toFun := fun f => ∑ i, f i • v i
                                /-
                                  α : Type u_1
                                  M : Type u_2
                                  R : Type u_3
                                  inst✝⁶ : Fintype α
                                  inst✝⁵ : Semiring R
                                  inst✝⁴ : AddCommMonoid M
                                  inst✝³ : Module R M
                                  S : Type u_4
                                  inst✝² : Semiring S
                                  inst✝¹ : Module S M
                                  inst✝ : SMulCommClass R S M
                                  v✝ v : α → M
                                  f g : α → R
                                  ⊢ Eq ((fun f => Finset.univ.sum fun i => HSMul.hSMul (f i) (v i)) (HAdd.hAdd f …
                                -/
      map_add' := fun f g => by simp_rw [← Finset.sum_add_distrib, ← add_smul]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                 /-
                                   α : Type u_1
                                   M : Type u_2
                                   R : Type u_3
                                   inst✝⁶ : Fintype α
                                   inst✝⁵ : Semiring R
                                   inst✝⁴ : AddCommMonoid M
                                   inst✝³ : Module R M
                                   S : Type u_4
                                   inst✝² : Semiring S
                                   inst✝¹ : Module S M
                                   inst✝ : SMulCommClass R S M
                                   v✝ v : α → M
                                   r : R
                                   f : α → R
                                   ⊢ Eq ({ toFun := fun f => Finset.univ.sum fun i => HSMul.hSMul (f i) (v i), ma …
                                 -/
      map_smul' := fun r f => by simp_rw [Finset.smul_sum, smul_smul]; rfl }
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                     /-
                       α : Type u_1
                       M : Type u_2
                       R : Type u_3
                       inst✝⁶ : Fintype α
                       inst✝⁵ : Semiring R
                       inst✝⁴ : AddCommMonoid M
                       inst✝³ : Module R M
                       S : Type u_4
                       inst✝² : Semiring S
                       inst✝¹ : Module S M
                       inst✝ : SMulCommClass R S M
                       v✝ u v : α → M
                       ⊢ Eq ((fun v => { toFun := fun f => Finset.univ.sum fun i => HSMul.hSMul (f i) …
                     -/
  map_add' u v := by ext; simp [Finset.sum_add_distrib, Pi.add_apply, smul_add]
                          /-
                            🎉 no goals
                          -/
                      /-
                        α : Type u_1
                        M : Type u_2
                        R : Type u_3
                        inst✝⁶ : Fintype α
                        inst✝⁵ : Semiring R
                        inst✝⁴ : AddCommMonoid M
                        inst✝³ : Module R M
                        S : Type u_4
                        inst✝² : Semiring S
                        inst✝¹ : Module S M
                        inst✝ : SMulCommClass R S M
                        v✝ : α → M
                        r : S
                        v : α → M
                        ⊢ Eq ({ toFun := fun v => { toFun := fun f => Finset.univ.sum fun i => HSMul.h …
                      -/
  map_smul' r v := by ext; simp [Finset.smul_sum, smul_comm]
                           /-
                             🎉 no goals
                           -/


@[deprecated (since := "2024-08-29")] alias Fintype.total := Fintype.linearCombination


theorem Fintype.linearCombination_apply (f) : Fintype.linearCombination R S v f = ∑ i, f i • v i :=
  rfl


@[deprecated (since := "2024-08-29")] alias Fintype.total_apply := Fintype.linearCombination_apply


@[simp]
theorem Fintype.linearCombination_apply_single [DecidableEq α] (i : α) (r : R) :
    Fintype.linearCombination R S v (Pi.single i r) = r • v i := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝⁷ : Fintype α
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    S : Type u_4
    inst✝³ : Semiring S
    inst✝² : Module S M
    inst✝¹ : SMulCommClass R S M
    v : α → M
    inst✝ : DecidableEq α
    i : α
    r : R
    ⊢ Eq (((Fintype.linearCombination R S) v) (Pi.single i r)) (HSMul.hSMul r (v i))
  -/
  simp_rw [Fintype.linearCombination_apply, Pi.single_apply, ite_smul, zero_smul]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝⁷ : Fintype α
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    S : Type u_4
    inst✝³ : Semiring S
    inst✝² : Module S M
    inst✝¹ : SMulCommClass R S M
    v : α → M
    inst✝ : DecidableEq α
    i : α
    r : R
    ⊢ Eq (Finset.univ.sum fun x => ite (Eq x i) (HSMul.hSMul r (v x)) 0) (HSMul.hS …
  -/
  rw [Finset.sum_ite_eq', if_pos (Finset.mem_univ _)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias Fintype.total_apply_single :=
  Fintype.linearCombination_apply_single


theorem Finsupp.linearCombination_eq_fintype_linearCombination_apply (x : α → R) :
    linearCombination R v ((Finsupp.linearEquivFunOnFinite R R α).symm x) =
      Fintype.linearCombination R S v x := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝⁶ : Fintype α
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Semiring S
    inst✝¹ : Module S M
    inst✝ : SMulCommClass R S M
    v : α → M
    x : α → R
    ⊢ Eq ((Finsupp.linearCombination R v) ((Finsupp.linearEquivFunOnFinite R R α). …
  -/
  apply Finset.sum_subset
    /-
      case h
      α : Type u_1
      M : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype α
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      v : α → M
      x : α → R
      ⊢ HasSubset.Subset ((Finsupp.linearEquivFunOnFinite R R α).symm x).support Fin …
    -/
  · exact Finset.subset_univ _
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      M : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype α
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      v : α → M
      x : α → R
      ⊢ ∀ (x_1 : α), Membership.mem Finset.univ x_1 → Not (Membership.mem ((Finsupp. …
    -/
  · intro x _ hx
    /-
      case hf
      α : Type u_1
      M : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype α
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      v : α → M
      x✝ : α → R
      x : α
      a✝ : Membership.mem Finset.univ x
      hx : Not (Membership.mem ((Finsupp.linearEquivFunOnFinite R R α).symm x✝).supp …
      ⊢ Eq ((fun i => ⇑((fun i => LinearMap.id.smulRight (v i)) i)) x (((Finsupp.lin …
    -/
    rw [Finsupp.not_mem_support_iff.mp hx]
    /-
      case hf
      α : Type u_1
      M : Type u_2
      R : Type u_3
      inst✝⁶ : Fintype α
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      v : α → M
      x✝ : α → R
      x : α
      a✝ : Membership.mem Finset.univ x
      hx : Not (Membership.mem ((Finsupp.linearEquivFunOnFinite R R α).symm x✝).supp …
      ⊢ Eq ((fun i => ⇑((fun i => LinearMap.id.smulRight (v i)) i)) x 0) 0
    -/
    exact zero_smul _ _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias Finsupp.total_eq_fintype_total_apply :=
  Finsupp.linearCombination_eq_fintype_linearCombination_apply


theorem Finsupp.linearCombination_eq_fintype_linearCombination :
    (linearCombination R v).comp (Finsupp.linearEquivFunOnFinite R R α).symm.toLinearMap =
      Fintype.linearCombination R S v :=
  LinearMap.ext <| linearCombination_eq_fintype_linearCombination_apply R S v


@[deprecated (since := "2024-08-29")] alias Finsupp.total_eq_fintype_total :=
  Finsupp.linearCombination_eq_fintype_linearCombination


@[simp]
theorem Fintype.range_linearCombination :
    LinearMap.range (Fintype.linearCombination R S v) = Submodule.span R (Set.range v) := by
  rw [← Finsupp.linearCombination_eq_fintype_linearCombination, LinearMap.range_comp,
      LinearEquiv.range, Submodule.map_top, Finsupp.range_linearCombination]


@[deprecated (since := "2024-08-29")] alias Fintype.range_total := Fintype.range_linearCombination


/-- An element `x` lies in the span of `v` iff it can be written as sum `∑ cᵢ • vᵢ = x`.
-/
theorem mem_span_range_iff_exists_fun :
    x ∈ span R (range v) ↔ ∃ c : α → R, ∑ i, c i • v i = x := by
  -- Porting note: `Finsupp.equivFunOnFinite.surjective.exists` should be come before `simp`.
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝³ : Fintype α
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    x : M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.range v)) x) (Exists fun c => Eq  …
  -/
  rw [Finsupp.equivFunOnFinite.surjective.exists]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝³ : Fintype α
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    x : M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.range v)) x) (Exists fun x_1 => E …
  -/
  simp only [Finsupp.mem_span_range_iff_exists_finsupp, Finsupp.equivFunOnFinite_apply]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝³ : Fintype α
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    x : M
    ⊢ Iff (Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (v i)) x) (Exists fu …
  -/
  exact exists_congr fun c => Eq.congr_left <| Finsupp.sum_fintype _ _ fun i => zero_smul _ _
  /-
    🎉 no goals
  -/


/-- A family `v : α → V` is generating `V` iff every element `(x : V)`
can be written as sum `∑ cᵢ • vᵢ = x`.
-/
theorem top_le_span_range_iff_forall_exists_fun :
    ⊤ ≤ span R (range v) ↔ ∀ x, ∃ c : α → R, ∑ i, c i • v i = x := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝³ : Fintype α
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    ⊢ Iff (LE.le Top.top (Submodule.span R (Set.range v))) (∀ (x : M), Exists fun  …
  -/
  simp_rw [← mem_span_range_iff_exists_fun]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝³ : Fintype α
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : α → M
    ⊢ Iff (LE.le Top.top (Submodule.span R (Set.range v))) (∀ (x : M), Membership. …
  -/
  exact ⟨fun h x => h trivial, fun h x _ => h x⟩
  /-
    🎉 no goals
  -/


/-- Pick some representation of `x : span R w` as a linear combination in `w`,
  ((Finsupp.mem_span_iff_linearCombination _ _ _).mp x.2).choose
-/
irreducible_def Span.repr (w : Set M) (x : span R w) : w →₀ R :=
  ((Finsupp.mem_span_iff_linearCombination _ _ _).mp x.2).choose


@[simp]
theorem Span.finsupp_linearCombination_repr {w : Set M} (x : span R w) :
    Finsupp.linearCombination R ((↑) : w → M) (Span.repr R w x) = x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    w : Set M
    x : Subtype fun x => Membership.mem (Submodule.span R w) x
    ⊢ Eq ((Finsupp.linearCombination R Subtype.val) (Span.repr R w x)) ↑x
  -/
  rw [Span.repr_def]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    w : Set M
    x : Subtype fun x => Membership.mem (Submodule.span R w) x
    ⊢ Eq ((Finsupp.linearCombination R Subtype.val) ⋯.choose) ↑x
  -/
  exact ((Finsupp.mem_span_iff_linearCombination _ _ _).mp x.2).choose_spec
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias Span.finsupp_total_repr :=
  Span.finsupp_linearCombination_repr

theorem LinearMap.map_finsupp_linearCombination (f : M →ₗ[R] N) {ι : Type*} {g : ι → M}
    (l : ι →₀ R) : f (linearCombination R g l) = linearCombination R (f ∘ g) l :=
  apply_linearCombination _ _ _ _


@[deprecated (since := "2024-08-29")] alias LinearMap.map_finsupp_total :=
  LinearMap.map_finsupp_linearCombination


theorem mem_span_finset {s : Finset M} {x : M} :
    x ∈ span R (↑s : Set M) ↔ ∃ f : M → R, ∑ i ∈ s, f i • i = x :=
  ⟨fun hx =>
    let ⟨v, hvs, hvx⟩ :=
      (Finsupp.mem_span_image_iff_linearCombination _).1
                                                        /-
                                                          R : Type u_1
                                                          M : Type u_2
                                                          inst✝² : Semiring R
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          s : Finset M
                                                          x : M
                                                          hx : Membership.mem (Submodule.span R ↑s) x
                                                          ⊢ Membership.mem (Submodule.span R (Set.image id ↑s)) x
                                                        -/
        (show x ∈ span R (_root_.id '' (↑s : Set M)) by rwa [Set.image_id])
                                                        /-
                                                          🎉 no goals
                                                        -/
    ⟨v, hvx ▸ (linearCombination_apply_of_mem_supported _ hvs).symm⟩,
    fun ⟨_, hf⟩ => hf ▸ sum_mem fun _ hi => smul_mem _ _ <| subset_span hi⟩


/-- An element `m ∈ M` is contained in the `R`-submodule spanned by a set `s ⊆ M`, if and only if
`m` can be written as a finite `R`-linear combination of elements of `s`.
The implementation uses `Finsupp.sum`. -/
theorem mem_span_set {m : M} {s : Set M} :
    m ∈ Submodule.span R s ↔
      ∃ c : M →₀ R, (c.support : Set M) ⊆ s ∧ (c.sum fun mi r => r • mi) = m := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    s : Set M
    ⊢ Iff (Membership.mem (Submodule.span R s) m) (Exists fun c => And (HasSubset. …
  -/
  conv_lhs => rw [← Set.image_id s]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    s : Set M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.image id s)) m) (Exists fun c =>  …
  -/
  exact Finsupp.mem_span_image_iff_linearCombination R (v := _root_.id (α := M))
  /-
    🎉 no goals
  -/


/-- An element `m ∈ M` is contained in the `R`-submodule spanned by a set `s ⊆ M`, if and only if
`m` can be written as a finite `R`-linear combination of elements of `s`.
The implementation uses a sum indexed by `Fin n` for some `n`. -/
lemma mem_span_set' {m : M} {s : Set M} :
    m ∈ Submodule.span R s ↔ ∃ (n : ℕ) (f : Fin n → R) (g : Fin n → s),
      ∑ i, f i • (g i : M) = m := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    s : Set M
    ⊢ Iff (Membership.mem (Submodule.span R s) m) (Exists fun n => Exists fun f => …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      m : M
      s : Set M
      h : Membership.mem (Submodule.span R s) m
      ⊢ Exists fun n => Exists fun f => Exists fun g => Eq (Finset.univ.sum fun i => …
    -/
  · rcases mem_span_set.1 h with ⟨c, cs, rfl⟩
    /-
      case refine_1.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      c : Finsupp M R
      cs : HasSubset.Subset (↑c.support) s
      h : Membership.mem (Submodule.span R s) (c.sum fun mi r => HSMul.hSMul r mi)
      ⊢ Exists fun n => Exists fun f => Exists fun g => Eq (Finset.univ.sum fun i => …
    -/
    have A : c.support ≃ Fin c.support.card := Finset.equivFin _
    /-
      case refine_1.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      c : Finsupp M R
      cs : HasSubset.Subset (↑c.support) s
      h : Membership.mem (Submodule.span R s) (c.sum fun mi r => HSMul.hSMul r mi)
      A : Equiv (Subtype fun x => Membership.mem c.support x) (Fin c.support.card)
      ⊢ Exists fun n => Exists fun f => Exists fun g => Eq (Finset.univ.sum fun i => …
    -/
    refine ⟨_, fun i ↦ c (A.symm i), fun i ↦ ⟨A.symm i, cs (A.symm i).2⟩, ?_⟩
    /-
      case refine_1.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      c : Finsupp M R
      cs : HasSubset.Subset (↑c.support) s
      h : Membership.mem (Submodule.span R s) (c.sum fun mi r => HSMul.hSMul r mi)
      A : Equiv (Subtype fun x => Membership.mem c.support x) (Fin c.support.card)
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((fun i => c ↑(A.symm i)) i) ↑((fun …
    -/
    rw [Finsupp.sum, ← Finset.sum_coe_sort c.support]
    /-
      case refine_1.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      c : Finsupp M R
      cs : HasSubset.Subset (↑c.support) s
      h : Membership.mem (Submodule.span R s) (c.sum fun mi r => HSMul.hSMul r mi)
      A : Equiv (Subtype fun x => Membership.mem c.support x) (Fin c.support.card)
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((fun i => c ↑(A.symm i)) i) ↑((fun …
    -/
    exact Fintype.sum_equiv A.symm _ (fun j ↦ c j • (j : M)) (fun i ↦ rfl)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      m : M
      s : Set M
      ⊢ (Exists fun n => Exists fun f => Exists fun g => Eq (Finset.univ.sum fun i = …
    -/
  · rintro ⟨n, f, g, rfl⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      n : Nat
      f : Fin n → R
      g : Fin n → ↑s
      ⊢ Membership.mem (Submodule.span R s) (Finset.univ.sum fun i => HSMul.hSMul (f …
    -/
    exact Submodule.sum_mem _ (fun i _ ↦ Submodule.smul_mem _ _ (Submodule.subset_span (g i).2))
    /-
      🎉 no goals
    -/


/-- The span of a subset `s` is the union over all `n` of the set of linear combinations of at most
`n` terms belonging to `s`. -/
lemma span_eq_iUnion_nat (s : Set M) :
    (Submodule.span R s : Set M) = ⋃ (n : ℕ),
      (fun (f : Fin n → (R × M)) ↦ ∑ i, (f i).1 • (f i).2) '' ({f | ∀ i, (f i).2 ∈ s}) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    ⊢ Eq (↑(Submodule.span R s)) (Set.iUnion fun n => Set.image (fun f => Finset.u …
  -/
  ext m
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    m : M
    ⊢ Iff (Membership.mem (↑(Submodule.span R s)) m) (Membership.mem (Set.iUnion f …
  -/
  simp only [SetLike.mem_coe, mem_iUnion, mem_image, mem_setOf_eq, mem_span_set']
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    m : M
    ⊢ Iff (Exists fun n => Exists fun f => Exists fun g => Eq (Finset.univ.sum fun …
  -/
  refine exists_congr (fun n ↦ ⟨?_, ?_⟩)
    /-
      case h.refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      m : M
      n : Nat
      ⊢ (Exists fun f => Exists fun g => Eq (Finset.univ.sum fun i => HSMul.hSMul (f …
    -/
  · rintro ⟨f, g, rfl⟩
    /-
      case h.refine_1.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      n : Nat
      f : Fin n → R
      g : Fin n → ↑s
      ⊢ Exists fun x => And (∀ (i : Fin n), Membership.mem s (x i).2) (Eq (Finset.un …
    -/
    exact ⟨fun i ↦ (f i, g i), fun i ↦ (g i).2, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      m : M
      n : Nat
      ⊢ (Exists fun x => And (∀ (i : Fin n), Membership.mem s (x i).2) (Eq (Finset.u …
    -/
  · rintro ⟨f, hf, rfl⟩
    /-
      case h.refine_2.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      n : Nat
      f : Fin n → Prod R M
      hf : ∀ (i : Fin n), Membership.mem s (f i).2
      ⊢ Exists fun f_1 => Exists fun g => Eq (Finset.univ.sum fun i => HSMul.hSMul ( …
    -/
    exact ⟨fun i ↦ (f i).1, fun i ↦ ⟨(f i).2, (hf i)⟩, rfl⟩
    /-
      🎉 no goals
    -/


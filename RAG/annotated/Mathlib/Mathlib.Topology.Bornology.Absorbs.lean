/-- A set `s` absorbs another set `t` if `t` is contained in all scalings of `s`
by all but a bounded set of elements. -/
def Absorbs (s t : Set α) : Prop :=
  ∀ᶠ a in cobounded M, t ⊆ a • s


/-- A set is *absorbent* if it absorbs every singleton. -/
def Absorbent (s : Set α) : Prop :=
  ∀ x, Absorbs M s {x}


                                            /-
                                              M : Type u_1
                                              α : Type u_2
                                              inst✝¹ : Bornology M
                                              inst✝ : SMul M α
                                              s : Set α
                                              ⊢ Absorbs M s EmptyCollection.emptyCollection
                                            -/
protected lemma empty : Absorbs M s ∅ := by simp [Absorbs]
                                            /-
                                              🎉 no goals
                                            -/


protected lemma eventually (h : Absorbs M s t) : ∀ᶠ a in cobounded M, t ⊆ a • s := h


                                                                     /-
                                                                       M : Type u_1
                                                                       α : Type u_2
                                                                       inst✝² : Bornology M
                                                                       inst✝¹ : SMul M α
                                                                       s t : Set α
                                                                       inst✝ : BoundedSpace M
                                                                       ⊢ Absorbs M s t
                                                                     -/
@[simp] lemma of_boundedSpace [BoundedSpace M] : Absorbs M s t := by simp [Absorbs]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma mono_left (h : Absorbs M s₁ t) (hs : s₁ ⊆ s₂) : Absorbs M s₂ t :=
  h.mono fun _a ha ↦ ha.trans <| smul_set_mono hs


lemma mono_right (h : Absorbs M s t₁) (ht : t₂ ⊆ t₁) : Absorbs M s t₂ :=
  h.mono fun _ ↦ ht.trans


lemma mono (h : Absorbs M s₁ t₁) (hs : s₁ ⊆ s₂) (ht : t₂ ⊆ t₁) : Absorbs M s₂ t₂ :=
  (h.mono_left hs).mono_right ht


@[simp]
lemma _root_.absorbs_union : Absorbs M s (t₁ ∪ t₂) ↔ Absorbs M s t₁ ∧ Absorbs M s t₂ := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Bornology M
    inst✝ : SMul M α
    s t₁ t₂ : Set α
    ⊢ Iff (Absorbs M s (Union.union t₁ t₂)) (And (Absorbs M s t₁) (Absorbs M s t₂))
  -/
  simp [Absorbs]
  /-
    🎉 no goals
  -/


protected lemma union (h₁ : Absorbs M s t₁) (h₂ : Absorbs M s t₂) : Absorbs M s (t₁ ∪ t₂) :=
  absorbs_union.2 ⟨h₁, h₂⟩


lemma _root_.Set.Finite.absorbs_sUnion {T : Set (Set α)} (hT : T.Finite) :
    Absorbs M s (⋃₀ T) ↔ ∀ t ∈ T, Absorbs M s t := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Bornology M
    inst✝ : SMul M α
    s : Set α
    T : Set (Set α)
    hT : T.Finite
    ⊢ Iff (Absorbs M s T.sUnion) (∀ (t : Set α), Membership.mem T t → Absorbs M s t)
  -/
  simp [Absorbs, hT]
  /-
    🎉 no goals
  -/


protected lemma sUnion (hT : T.Finite) (hs : ∀ t ∈ T, Absorbs M s t) :
    Absorbs M s (⋃₀ T) :=
  hT.absorbs_sUnion.2 hs


@[simp]
lemma _root_.absorbs_iUnion {ι : Sort*} [Finite ι] {t : ι → Set α} :
    Absorbs M s (⋃ i, t i) ↔ ∀ i, Absorbs M s (t i) :=
  (finite_range t).absorbs_sUnion.trans forall_mem_range


protected alias ⟨_, iUnion⟩ := absorbs_iUnion


lemma _root_.Set.Finite.absorbs_biUnion {ι : Type*} {t : ι → Set α} {I : Set ι} (hI : I.Finite) :
    Absorbs M s (⋃ i ∈ I, t i) ↔ ∀ i ∈ I, Absorbs M s (t i) := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Bornology M
    inst✝ : SMul M α
    s : Set α
    ι : Type u_3
    t : ι → Set α
    I : Set ι
    hI : I.Finite
    ⊢ Iff (Absorbs M s (Set.iUnion fun i => Set.iUnion fun h => t i)) (∀ (i : ι),  …
  -/
  simp [Absorbs, hI]
  /-
    🎉 no goals
  -/


protected alias ⟨_, biUnion⟩ := Set.Finite.absorbs_biUnion


@[simp]
lemma _root_.absorbs_biUnion_finset {ι : Type*} {t : ι → Set α} {I : Finset ι} :
    Absorbs M s (⋃ i ∈ I, t i) ↔ ∀ i ∈ I, Absorbs M s (t i) :=
  I.finite_toSet.absorbs_biUnion


protected alias ⟨_, biUnion_finset⟩ := absorbs_biUnion_finset


protected lemma add [AddZeroClass E] [DistribSMul M E]
    (h₁ : Absorbs M s₁ t₁) (h₂ : Absorbs M s₂ t₂) : Absorbs M (s₁ + s₂) (t₁ + t₂) :=
                                                 /-
                                                   M : Type u_1
                                                   E : Type u_2
                                                   inst✝² : Bornology M
                                                   s₁ s₂ t₁ t₂ : Set E
                                                   inst✝¹ : AddZeroClass E
                                                   inst✝ : DistribSMul M E
                                                   h₁ : Absorbs M s₁ t₁
                                                   h₂ : Absorbs M s₂ t₂
                                                   x : M
                                                   hx₁ : HasSubset.Subset t₁ (HSMul.hSMul x s₁)
                                                   hx₂ : HasSubset.Subset t₂ (HSMul.hSMul x s₂)
                                                   ⊢ HasSubset.Subset (HAdd.hAdd t₁ t₂) (HSMul.hSMul x (HAdd.hAdd s₁ s₂))
                                                 -/
  h₂.mp <| h₁.eventually.mono fun x hx₁ hx₂ ↦ by rw [smul_add]; exact add_subset_add hx₁ hx₂
                                                                /-
                                                                  🎉 no goals
                                                                -/


protected lemma zero [Zero E] [SMulZeroClass M E] {s : Set E} (hs : 0 ∈ s) : Absorbs M s 0 :=
  Eventually.of_forall fun _ ↦ zero_subset.2 <| zero_mem_smul_set hs


@[simp]
protected lemma Absorbs.univ : Absorbs G₀ univ s :=
                                                 /-
                                                   G₀ : Type u_1
                                                   α : Type u_2
                                                   inst✝² : GroupWithZero G₀
                                                   inst✝¹ : Bornology G₀
                                                   inst✝ : MulAction G₀ α
                                                   s : Set α
                                                   a : G₀
                                                   ha : Ne a 0
                                                   ⊢ HasSubset.Subset s (HSMul.hSMul a Set.univ)
                                                 -/
  (eventually_ne_cobounded 0).mono fun a ha ↦ by rw [smul_set_univ₀ ha]; apply subset_univ
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma absorbs_iff_eventually_cobounded_mapsTo :
    Absorbs G₀ s t ↔ ∀ᶠ c in cobounded G₀, MapsTo (c⁻¹ • ·) t s :=
  eventually_congr <| (eventually_ne_cobounded 0).mono fun c hc ↦ by
    /-
      G₀ : Type u_1
      α : Type u_2
      inst✝² : GroupWithZero G₀
      inst✝¹ : Bornology G₀
      inst✝ : MulAction G₀ α
      s t : Set α
      c : G₀
      hc : Ne c 0
      ⊢ Iff (HasSubset.Subset t (HSMul.hSMul c s)) (Set.MapsTo (fun x => HSMul.hSMul …
    -/
    rw [← preimage_smul_inv₀ hc]; rfl
                                  /-
                                    🎉 no goals
                                  -/


alias ⟨eventually_cobounded_mapsTo, _⟩ := absorbs_iff_eventually_cobounded_mapsTo


@[simp]
lemma absorbs_inter : Absorbs G₀ (s ∩ t) u ↔ Absorbs G₀ s u ∧ Absorbs G₀ t u := by
  /-
    G₀ : Type u_1
    α : Type u_2
    inst✝² : GroupWithZero G₀
    inst✝¹ : Bornology G₀
    inst✝ : MulAction G₀ α
    s t u : Set α
    ⊢ Iff (Absorbs G₀ (Inter.inter s t) u) (And (Absorbs G₀ s u) (Absorbs G₀ t u))
  -/
  simp only [absorbs_iff_eventually_cobounded_mapsTo, mapsTo_inter, eventually_and]
  /-
    🎉 no goals
  -/


protected lemma Absorbs.inter (hs : Absorbs G₀ s u) (ht : Absorbs G₀ t u) : Absorbs G₀ (s ∩ t) u :=
  absorbs_inter.2 ⟨hs, ht⟩


variable (G₀ u) in
/-- The filter of sets that absorb `u`. -/
def Filter.absorbing : Filter α where
  sets := {s | Absorbs G₀ s u}
  univ_sets := .univ
  sets_of_superset h := h.mono_left
  inter_sets := .inter


@[simp]
lemma Filter.mem_absorbing : s ∈ absorbing G₀ u ↔ Absorbs G₀ s u := .rfl


lemma Set.Finite.absorbs_sInter (hS : S.Finite) :
    Absorbs G₀ (⋂₀ S) t ↔ ∀ s ∈ S, Absorbs G₀ s t :=
  sInter_mem (f := absorbing G₀ t) hS


protected alias ⟨_, Absorbs.sInter⟩ := Set.Finite.absorbs_sInter


@[simp]
lemma absorbs_iInter {ι : Sort*} [Finite ι] {s : ι → Set α} :
    Absorbs G₀ (⋂ i, s i) t ↔ ∀ i, Absorbs G₀ (s i) t :=
  iInter_mem (f := absorbing G₀ t)


protected alias ⟨_, Absorbs.iInter⟩ := absorbs_iInter


lemma Set.Finite.absorbs_biInter {ι : Type*} {I : Set ι} (hI : I.Finite) {s : ι → Set α} :
    Absorbs G₀ (⋂ i ∈ I, s i) t ↔ ∀ i ∈ I, Absorbs G₀ (s i) t :=
  biInter_mem (f := absorbing G₀ t) hI


protected alias ⟨_, Absorbs.biInter⟩ := Set.Finite.absorbs_biInter


@[simp]
lemma absorbs_zero_iff [NeBot (cobounded G₀)]
    {E : Type*} [AddMonoid E] [DistribMulAction G₀ E] {s : Set E} :
    Absorbs G₀ s 0 ↔ 0 ∈ s := by
  simp only [absorbs_iff_eventually_cobounded_mapsTo, ← singleton_zero,
    mapsTo_singleton, smul_zero, eventually_const]


@[simp]
                                                                                /-
                                                                                  M : Type u_1
                                                                                  E : Type u_2
                                                                                  inst✝³ : Monoid M
                                                                                  inst✝² : AddGroup E
                                                                                  inst✝¹ : DistribMulAction M E
                                                                                  inst✝ : Bornology M
                                                                                  s t : Set E
                                                                                  ⊢ Iff (Absorbs M (Neg.neg s) (Neg.neg t)) (Absorbs M s t)
                                                                                -/
lemma absorbs_neg_neg {s t : Set E} : Absorbs M (-s) (-t) ↔ Absorbs M s t := by simp [Absorbs]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


alias ⟨Absorbs.of_neg_neg, Absorbs.neg_neg⟩ := absorbs_neg_neg


lemma Absorbs.sub {s₁ s₂ t₁ t₂ : Set E} (h₁ : Absorbs M s₁ t₁) (h₂ : Absorbs M s₂ t₂) :
    Absorbs M (s₁ - s₂) (t₁ - t₂) := by
  /-
    M : Type u_1
    E : Type u_2
    inst✝³ : Monoid M
    inst✝² : AddGroup E
    inst✝¹ : DistribMulAction M E
    inst✝ : Bornology M
    s₁ s₂ t₁ t₂ : Set E
    h₁ : Absorbs M s₁ t₁
    h₂ : Absorbs M s₂ t₂
    ⊢ Absorbs M (HSub.hSub s₁ s₂) (HSub.hSub t₁ t₂)
  -/
  simpa only [sub_eq_add_neg] using h₁.add h₂.neg_neg
  /-
    🎉 no goals
  -/


protected theorem mono (ht : Absorbent M s) (hsub : s ⊆ t) : Absorbent M t := fun x ↦
  (ht x).mono_left hsub


theorem _root_.absorbent_iff_forall_absorbs_singleton : Absorbent M s ↔ ∀ x, Absorbs M s {x} := .rfl


protected theorem absorbs (hs : Absorbent M s) {x : α} : Absorbs M s {x} := hs x


theorem absorbs_finite (hs : Absorbent M s) (ht : t.Finite) : Absorbs M s t := by
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Bornology M
    inst✝ : SMul M α
    s t : Set α
    hs : Absorbent M s
    ht : t.Finite
    ⊢ Absorbs M s t
  -/
  rw [← Set.biUnion_of_singleton t]
  /-
    M : Type u_1
    α : Type u_2
    inst✝¹ : Bornology M
    inst✝ : SMul M α
    s t : Set α
    hs : Absorbent M s
    ht : t.Finite
    ⊢ Absorbs M s (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton x)
  -/
  exact .biUnion ht fun _ _ => hs.absorbs
  /-
    🎉 no goals
  -/


theorem vadd_absorbs {M E : Type*} [Bornology M] [AddZeroClass E] [DistribSMul M E]
    {s₁ s₂ t : Set E} {x : E} (h₁ : Absorbent M s₁) (h₂ : Absorbs M s₂ t) :
    Absorbs M (s₁ + s₂) (x +ᵥ t) := by
  /-
    M : Type u_1
    E : Type u_2
    inst✝² : Bornology M
    inst✝¹ : AddZeroClass E
    inst✝ : DistribSMul M E
    s₁ s₂ t : Set E
    x : E
    h₁ : Absorbent M s₁
    h₂ : Absorbs M s₂ t
    ⊢ Absorbs M (HAdd.hAdd s₁ s₂) (HVAdd.hVAdd x t)
  -/
  rw [← singleton_vadd]; exact (h₁ x).add h₂
                         /-
                           🎉 no goals
                         -/


lemma absorbent_univ : Absorbent G₀ (univ : Set α) := fun _ ↦ .univ


lemma absorbent_iff_inv_smul {s : Set α} :
    Absorbent G₀ s ↔ ∀ x, ∀ᶠ c in cobounded G₀, c⁻¹ • x ∈ s :=
                           /-
                             G₀ : Type u_1
                             α : Type u_2
                             inst✝² : GroupWithZero G₀
                             inst✝¹ : Bornology G₀
                             inst✝ : MulAction G₀ α
                             s : Set α
                             x : α
                             ⊢ Iff (Absorbs G₀ s (Singleton.singleton x)) (Filter.Eventually (fun c => Memb …
                           -/
  forall_congr' fun x ↦ by simp only [absorbs_iff_eventually_cobounded_mapsTo, mapsTo_singleton]
                           /-
                             🎉 no goals
                           -/


lemma Absorbent.zero_mem [NeBot (cobounded G₀)] [AddMonoid E] [DistribMulAction G₀ E]
    {s : Set E} (hs : Absorbent G₀ s) : (0 : E) ∈ s :=
  absorbs_zero_iff.1 (hs 0)


protected theorem Absorbs.restrict_scalars
    {M N α : Type*} [Monoid N] [SMul M N] [SMul M α] [MulAction N α]
    [IsScalarTower M N α] [Bornology M] [Bornology N] {s t : Set α} (h : Absorbs N s t)
    (hbdd : Tendsto (· • 1 : M → N) (cobounded M) (cobounded N)) :
    Absorbs M s t :=
                                            /-
                                              M : Type u_1
                                              N : Type u_2
                                              α : Type u_3
                                              inst✝⁶ : Monoid N
                                              inst✝⁵ : SMul M N
                                              inst✝⁴ : SMul M α
                                              inst✝³ : MulAction N α
                                              inst✝² : IsScalarTower M N α
                                              inst✝¹ : Bornology M
                                              inst✝ : Bornology N
                                              s t : Set α
                                              h : Absorbs N s t
                                              hbdd : Filter.Tendsto (fun x => HSMul.hSMul x 1) (Bornology.cobounded M) (Born …
                                              x : M
                                              hx : HasSubset.Subset t (HSMul.hSMul (HSMul.hSMul x 1) s)
                                              ⊢ HasSubset.Subset t (HSMul.hSMul x s)
                                            -/
  (hbdd.eventually h).mono <| fun x hx ↦ by rwa [smul_one_smul N x s] at hx
                                            /-
                                              🎉 no goals
                                            -/


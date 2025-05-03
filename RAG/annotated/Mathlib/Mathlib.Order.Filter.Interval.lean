/-- A pair of filters `l₁`, `l₂` has `TendstoIxxClass Ixx` property if `Ixx a b` tends to
`l₂.small_sets` as `a` and `b` tend to `l₁`. In all instances `Ixx` is one of `Set.Icc`, `Set.Ico`,
`Set.Ioc`, or `Set.Ioo`. The instances provide the best `l₂` for a given `l₁`. In many cases
`l₁ = l₂` but sometimes we can drop an endpoint from an interval: e.g., we prove
`TendstoIxxClass Set.Ico (𝓟 (Set.Iic a)) (𝓟 (Set.Iio a))`, i.e., if `u₁ n` and `u₂ n` belong
eventually to `Set.Iic a`, then the interval `Set.Ico (u₁ n) (u₂ n)` is eventually included in
`Set.Iio a`.

We mark `l₂` as an `outParam` so that Lean can automatically find an appropriate `l₂` based on
`Ixx` and `l₁`. This way, e.g., `tendsto.Ico h₁ h₂` works without specifying explicitly `l₂`. -/
class TendstoIxxClass (Ixx : α → α → Set α) (l₁ : Filter α) (l₂ : outParam <| Filter α) : Prop where
  /-- `Function.uncurry Ixx` tends to `l₂.smallSets` along `l₁ ×ˢ l₁`. In other words, for any
  `s ∈ l₂` there exists `t ∈ l₁` such that `Ixx x y ⊆ s` whenever `x ∈ t` and `y ∈ t`.

  Use lemmas like `Filter.Tendsto.Icc` instead. -/
  tendsto_Ixx : Tendsto (fun p : α × α => Ixx p.1 p.2) (l₁ ×ˢ l₁) l₂.smallSets


theorem tendstoIxxClass_principal {s t : Set α} {Ixx : α → α → Set α} :
    TendstoIxxClass Ixx (𝓟 s) (𝓟 t) ↔ ∀ᵉ (x ∈ s) (y ∈ s), Ixx x y ⊆ t :=
  Iff.trans ⟨fun h => h.1, fun h => ⟨h⟩⟩ <| by
    simp only [smallSets_principal, prod_principal_principal, tendsto_principal_principal,
      forall_prod_set, mem_powerset_iff, mem_principal]


theorem tendstoIxxClass_inf {l₁ l₁' l₂ l₂' : Filter α} {Ixx} [h : TendstoIxxClass Ixx l₁ l₂]
    [h' : TendstoIxxClass Ixx l₁' l₂'] : TendstoIxxClass Ixx (l₁ ⊓ l₁') (l₂ ⊓ l₂') :=
      /-
        α : Type u_1
        l₁ l₁' l₂ l₂' : Filter α
        Ixx : α → α → Set α
        h : Filter.TendstoIxxClass Ixx l₁ l₂
        h' : Filter.TendstoIxxClass Ixx l₁' l₂'
        ⊢ Filter.Tendsto (fun p => Ixx p.1 p.2) (SProd.sprod (Min.min l₁ l₁') (Min.min …
      -/
  ⟨by simpa only [prod_inf_prod, smallSets_inf] using h.1.inf h'.1⟩
      /-
        🎉 no goals
      -/


theorem tendstoIxxClass_of_subset {l₁ l₂ : Filter α} {Ixx Ixx' : α → α → Set α}
    (h : ∀ a b, Ixx a b ⊆ Ixx' a b) [h' : TendstoIxxClass Ixx' l₁ l₂] : TendstoIxxClass Ixx l₁ l₂ :=
  ⟨h'.1.smallSets_mono <| Eventually.of_forall <| Prod.forall.2 h⟩


theorem HasBasis.tendstoIxxClass {ι : Type*} {p : ι → Prop} {s} {l : Filter α}
    (hl : l.HasBasis p s) {Ixx : α → α → Set α}
    (H : ∀ i, p i → ∀ x ∈ s i, ∀ y ∈ s i, Ixx x y ⊆ s i) : TendstoIxxClass Ixx l l :=
  ⟨(hl.prod_self.tendsto_iff hl.smallSets).2 fun i hi => ⟨i, hi, fun _ h => H i hi _ h.1 _ h.2⟩⟩


protected theorem Tendsto.Icc {l₁ l₂ : Filter α} [TendstoIxxClass Icc l₁ l₂] {lb : Filter β}
    {u₁ u₂ : β → α} (h₁ : Tendsto u₁ lb l₁) (h₂ : Tendsto u₂ lb l₁) :
    Tendsto (fun x => Icc (u₁ x) (u₂ x)) lb l₂.smallSets :=
  (@TendstoIxxClass.tendsto_Ixx α Set.Icc _ _ _).comp <| h₁.prod_mk h₂


protected theorem Tendsto.Ioc {l₁ l₂ : Filter α} [TendstoIxxClass Ioc l₁ l₂] {lb : Filter β}
    {u₁ u₂ : β → α} (h₁ : Tendsto u₁ lb l₁) (h₂ : Tendsto u₂ lb l₁) :
    Tendsto (fun x => Ioc (u₁ x) (u₂ x)) lb l₂.smallSets :=
  (@TendstoIxxClass.tendsto_Ixx α Set.Ioc _ _ _).comp <| h₁.prod_mk h₂


protected theorem Tendsto.Ico {l₁ l₂ : Filter α} [TendstoIxxClass Ico l₁ l₂] {lb : Filter β}
    {u₁ u₂ : β → α} (h₁ : Tendsto u₁ lb l₁) (h₂ : Tendsto u₂ lb l₁) :
    Tendsto (fun x => Ico (u₁ x) (u₂ x)) lb l₂.smallSets :=
  (@TendstoIxxClass.tendsto_Ixx α Set.Ico _ _ _).comp <| h₁.prod_mk h₂


protected theorem Tendsto.Ioo {l₁ l₂ : Filter α} [TendstoIxxClass Ioo l₁ l₂] {lb : Filter β}
    {u₁ u₂ : β → α} (h₁ : Tendsto u₁ lb l₁) (h₂ : Tendsto u₂ lb l₁) :
    Tendsto (fun x => Ioo (u₁ x) (u₂ x)) lb l₂.smallSets :=
  (@TendstoIxxClass.tendsto_Ixx α Set.Ioo _ _ _).comp <| h₁.prod_mk h₂



instance tendsto_Icc_atTop_atTop : TendstoIxxClass Icc (atTop : Filter α) atTop :=
  (hasBasis_iInf_principal_finite _).tendstoIxxClass fun _ _ =>
    Set.OrdConnected.out <| ordConnected_biInter fun _ _ => ordConnected_Ici


instance tendsto_Ico_atTop_atTop : TendstoIxxClass Ico (atTop : Filter α) atTop :=
  tendstoIxxClass_of_subset fun _ _ => Ico_subset_Icc_self


instance tendsto_Ioc_atTop_atTop : TendstoIxxClass Ioc (atTop : Filter α) atTop :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Ioo_atTop_atTop : TendstoIxxClass Ioo (atTop : Filter α) atTop :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Icc_self


instance tendsto_Icc_atBot_atBot : TendstoIxxClass Icc (atBot : Filter α) atBot :=
  (hasBasis_iInf_principal_finite _).tendstoIxxClass fun _ _ =>
    Set.OrdConnected.out <| ordConnected_biInter fun _ _ => ordConnected_Iic


instance tendsto_Ico_atBot_atBot : TendstoIxxClass Ico (atBot : Filter α) atBot :=
  tendstoIxxClass_of_subset fun _ _ => Ico_subset_Icc_self


instance tendsto_Ioc_atBot_atBot : TendstoIxxClass Ioc (atBot : Filter α) atBot :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Ioo_atBot_atBot : TendstoIxxClass Ioo (atBot : Filter α) atBot :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Icc_self


instance OrdConnected.tendsto_Icc {s : Set α} [hs : OrdConnected s] :
    TendstoIxxClass Icc (𝓟 s) (𝓟 s) :=
  tendstoIxxClass_principal.2 hs.out


instance tendsto_Ico_Ici_Ici {a : α} : TendstoIxxClass Ico (𝓟 (Ici a)) (𝓟 (Ici a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ico_subset_Icc_self


instance tendsto_Ico_Ioi_Ioi {a : α} : TendstoIxxClass Ico (𝓟 (Ioi a)) (𝓟 (Ioi a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ico_subset_Icc_self


instance tendsto_Ico_Iic_Iio {a : α} : TendstoIxxClass Ico (𝓟 (Iic a)) (𝓟 (Iio a)) :=
  tendstoIxxClass_principal.2 fun _ _ _ h₁ _ h₂ => lt_of_lt_of_le h₂.2 h₁


instance tendsto_Ico_Iio_Iio {a : α} : TendstoIxxClass Ico (𝓟 (Iio a)) (𝓟 (Iio a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ico_subset_Icc_self


instance tendsto_Ioc_Ici_Ioi {a : α} : TendstoIxxClass Ioc (𝓟 (Ici a)) (𝓟 (Ioi a)) :=
  tendstoIxxClass_principal.2 fun _ h₁ _ _ _ h₂ => lt_of_le_of_lt h₁ h₂.1


instance tendsto_Ioc_Iic_Iic {a : α} : TendstoIxxClass Ioc (𝓟 (Iic a)) (𝓟 (Iic a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Ioc_Iio_Iio {a : α} : TendstoIxxClass Ioc (𝓟 (Iio a)) (𝓟 (Iio a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Ioc_Ioi_Ioi {a : α} : TendstoIxxClass Ioc (𝓟 (Ioi a)) (𝓟 (Ioi a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Ioo_Ici_Ioi {a : α} : TendstoIxxClass Ioo (𝓟 (Ici a)) (𝓟 (Ioi a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Ioc_self


instance tendsto_Ioo_Iic_Iio {a : α} : TendstoIxxClass Ioo (𝓟 (Iic a)) (𝓟 (Iio a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Ico_self


instance tendsto_Ioo_Ioi_Ioi {a : α} : TendstoIxxClass Ioo (𝓟 (Ioi a)) (𝓟 (Ioi a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Ioc_self


instance tendsto_Ioo_Iio_Iio {a : α} : TendstoIxxClass Ioo (𝓟 (Iio a)) (𝓟 (Iio a)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioo_subset_Ioc_self


instance tendsto_Icc_Icc_Icc {a b : α} : TendstoIxxClass Icc (𝓟 (Icc a b)) (𝓟 (Icc a b)) :=
  tendstoIxxClass_principal.mpr fun _x hx _y hy => Icc_subset_Icc hx.1 hy.2


instance tendsto_Ioc_Icc_Icc {a b : α} : TendstoIxxClass Ioc (𝓟 (Icc a b)) (𝓟 (Icc a b)) :=
  tendstoIxxClass_of_subset fun _ _ => Ioc_subset_Icc_self


instance tendsto_Icc_pure_pure {a : α} : TendstoIxxClass Icc (pure a) (pure a : Filter α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : PartialOrder α
    a : α
    ⊢ Filter.TendstoIxxClass Set.Icc (Pure.pure a) (Pure.pure a)
  -/
  rw [← principal_singleton]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : PartialOrder α
    a : α
    ⊢ Filter.TendstoIxxClass Set.Icc (Filter.principal (Singleton.singleton a)) (F …
  -/
  exact tendstoIxxClass_principal.2 ordConnected_singleton.out
  /-
    🎉 no goals
  -/


instance tendsto_Ico_pure_bot {a : α} : TendstoIxxClass Ico (pure a) ⊥ :=
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        a : α
        ⊢ Filter.Tendsto (fun p => Set.Ico p.1 p.2) (SProd.sprod (Pure.pure a) (Pure.p …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance tendsto_Ioc_pure_bot {a : α} : TendstoIxxClass Ioc (pure a) ⊥ :=
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        a : α
        ⊢ Filter.Tendsto (fun p => Set.Ioc p.1 p.2) (SProd.sprod (Pure.pure a) (Pure.p …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance tendsto_Ioo_pure_bot {a : α} : TendstoIxxClass Ioo (pure a) ⊥ :=
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        a : α
        ⊢ Filter.Tendsto (fun p => Set.Ioo p.1 p.2) (SProd.sprod (Pure.pure a) (Pure.p …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance tendsto_Icc_uIcc_uIcc {a b : α} : TendstoIxxClass Icc (𝓟 [[a, b]]) (𝓟 [[a, b]]) :=
  Filter.tendsto_Icc_Icc_Icc


instance tendsto_Ioc_uIcc_uIcc {a b : α} : TendstoIxxClass Ioc (𝓟 [[a, b]]) (𝓟 [[a, b]]) :=
  Filter.tendsto_Ioc_Icc_Icc


instance tendsto_uIcc_of_Icc {l : Filter α} [TendstoIxxClass Icc l l] :
    TendstoIxxClass uIcc l l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    l : Filter α
    inst✝ : Filter.TendstoIxxClass Set.Icc l l
    ⊢ Filter.TendstoIxxClass Set.uIcc l l
  -/
  refine ⟨fun s hs => mem_map.2 <| mem_prod_self_iff.2 ?_⟩
  obtain ⟨t, htl, hts⟩ : ∃ t ∈ l, ∀ p ∈ (t : Set α) ×ˢ t, Icc (p : α × α).1 p.2 ∈ s :=
    mem_prod_self_iff.1 (mem_map.1 (tendsto_fst.Icc tendsto_snd hs))
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    l : Filter α
    inst✝ : Filter.TendstoIxxClass Set.Icc l l
    s : Set (Set α)
    hs : Membership.mem l.smallSets s
    t : Set α
    htl : Membership.mem l t
    hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
    ⊢ Exists fun t => And (Membership.mem l t) (HasSubset.Subset (SProd.sprod t t) …
  -/
  refine ⟨t, htl, fun p hp => ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    l : Filter α
    inst✝ : Filter.TendstoIxxClass Set.Icc l l
    s : Set (Set α)
    hs : Membership.mem l.smallSets s
    t : Set α
    htl : Membership.mem l t
    hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
    p : Prod α α
    hp : Membership.mem (SProd.sprod t t) p
    ⊢ Membership.mem (Set.preimage (fun p => Set.uIcc p.1 p.2) s) p
  -/
  rcases le_total p.1 p.2 with h | h
    /-
      case intro.intro.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      l : Filter α
      inst✝ : Filter.TendstoIxxClass Set.Icc l l
      s : Set (Set α)
      hs : Membership.mem l.smallSets s
      t : Set α
      htl : Membership.mem l t
      hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
      p : Prod α α
      hp : Membership.mem (SProd.sprod t t) p
      h : LE.le p.1 p.2
      ⊢ Membership.mem (Set.preimage (fun p => Set.uIcc p.1 p.2) s) p
    -/
  · rw [mem_preimage, uIcc_of_le h]
    /-
      case intro.intro.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      l : Filter α
      inst✝ : Filter.TendstoIxxClass Set.Icc l l
      s : Set (Set α)
      hs : Membership.mem l.smallSets s
      t : Set α
      htl : Membership.mem l t
      hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
      p : Prod α α
      hp : Membership.mem (SProd.sprod t t) p
      h : LE.le p.1 p.2
      ⊢ Membership.mem s (Set.Icc p.1 p.2)
    -/
    exact hts p hp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      l : Filter α
      inst✝ : Filter.TendstoIxxClass Set.Icc l l
      s : Set (Set α)
      hs : Membership.mem l.smallSets s
      t : Set α
      htl : Membership.mem l t
      hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
      p : Prod α α
      hp : Membership.mem (SProd.sprod t t) p
      h : LE.le p.2 p.1
      ⊢ Membership.mem (Set.preimage (fun p => Set.uIcc p.1 p.2) s) p
    -/
  · rw [mem_preimage, uIcc_of_ge h]
    /-
      case intro.intro.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      l : Filter α
      inst✝ : Filter.TendstoIxxClass Set.Icc l l
      s : Set (Set α)
      hs : Membership.mem l.smallSets s
      t : Set α
      htl : Membership.mem l t
      hts : ∀ (p : Prod α α), Membership.mem (SProd.sprod t t) p → Membership.mem s  …
      p : Prod α α
      hp : Membership.mem (SProd.sprod t t) p
      h : LE.le p.2 p.1
      ⊢ Membership.mem s (Set.Icc p.2 p.1)
    -/
    exact hts ⟨p.2, p.1⟩ ⟨hp.2, hp.1⟩
    /-
      🎉 no goals
    -/


protected theorem Tendsto.uIcc {l : Filter α} [TendstoIxxClass Icc l l] {f g : β → α}
    {lb : Filter β} (hf : Tendsto f lb l) (hg : Tendsto g lb l) :
    Tendsto (fun x => [[f x, g x]]) lb l.smallSets :=
  (@TendstoIxxClass.tendsto_Ixx α Set.uIcc _ _ _).comp <| hf.prod_mk hg



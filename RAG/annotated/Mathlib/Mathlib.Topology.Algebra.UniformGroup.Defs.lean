/-- A uniform group is a group in which multiplication and inversion are uniformly continuous. -/
class UniformGroup (α : Type*) [UniformSpace α] [Group α] : Prop where
  uniformContinuous_div : UniformContinuous fun p : α × α => p.1 / p.2


/-- A uniform additive group is an additive group in which addition
  and negation are uniformly continuous. -/
class UniformAddGroup (α : Type*) [UniformSpace α] [AddGroup α] : Prop where
  uniformContinuous_sub : UniformContinuous fun p : α × α => p.1 - p.2


@[to_additive]
theorem UniformGroup.mk' {α} [UniformSpace α] [Group α]
    (h₁ : UniformContinuous fun p : α × α => p.1 * p.2) (h₂ : UniformContinuous fun p : α => p⁻¹) :
    UniformGroup α :=
  ⟨by simpa only [div_eq_mul_inv] using
    h₁.comp (uniformContinuous_fst.prod_mk (h₂.comp uniformContinuous_snd))⟩


@[to_additive]
theorem uniformContinuous_div : UniformContinuous fun p : α × α => p.1 / p.2 :=
  UniformGroup.uniformContinuous_div


@[to_additive]
theorem UniformContinuous.div [UniformSpace β] {f : β → α} {g : β → α} (hf : UniformContinuous f)
    (hg : UniformContinuous g) : UniformContinuous fun x => f x / g x :=
  uniformContinuous_div.comp (hf.prod_mk hg)


@[to_additive]
theorem UniformContinuous.inv [UniformSpace β] {f : β → α} (hf : UniformContinuous f) :
    UniformContinuous fun x => (f x)⁻¹ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : UniformSpace β
    f : β → α
    hf : UniformContinuous f
    ⊢ UniformContinuous fun x => Inv.inv (f x)
  -/
  have : UniformContinuous fun x => 1 / f x := uniformContinuous_const.div hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : UniformSpace β
    f : β → α
    hf : UniformContinuous f
    this : UniformContinuous fun x => HDiv.hDiv 1 (f x)
    ⊢ UniformContinuous fun x => Inv.inv (f x)
  -/
  simp_all
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformContinuous_inv : UniformContinuous fun x : α => x⁻¹ :=
  uniformContinuous_id.inv


@[to_additive]
theorem UniformContinuous.mul [UniformSpace β] {f : β → α} {g : β → α} (hf : UniformContinuous f)
    (hg : UniformContinuous g) : UniformContinuous fun x => f x * g x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : UniformSpace β
    f g : β → α
    hf : UniformContinuous f
    hg : UniformContinuous g
    ⊢ UniformContinuous fun x => HMul.hMul (f x) (g x)
  -/
  have : UniformContinuous fun x => f x / (g x)⁻¹ := hf.div hg.inv
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : UniformSpace β
    f g : β → α
    hf : UniformContinuous f
    hg : UniformContinuous g
    this : UniformContinuous fun x => HDiv.hDiv (f x) (Inv.inv (g x))
    ⊢ UniformContinuous fun x => HMul.hMul (f x) (g x)
  -/
  simp_all
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformContinuous_mul : UniformContinuous fun p : α × α => p.1 * p.2 :=
  uniformContinuous_fst.mul uniformContinuous_snd


@[to_additive]
theorem UniformContinuous.mul_const [UniformSpace β] {f : β → α} (hf : UniformContinuous f)
    (a : α) : UniformContinuous fun x ↦ f x * a :=
  hf.mul uniformContinuous_const


@[to_additive]
theorem UniformContinuous.const_mul [UniformSpace β] {f : β → α} (hf : UniformContinuous f)
    (a : α) : UniformContinuous fun x ↦ a * f x :=
  uniformContinuous_const.mul hf


@[to_additive]
theorem uniformContinuous_mul_left (a : α) : UniformContinuous fun b : α => a * b :=
  uniformContinuous_id.const_mul _


@[to_additive]
theorem uniformContinuous_mul_right (a : α) : UniformContinuous fun b : α => b * a :=
  uniformContinuous_id.mul_const _


@[to_additive]
theorem UniformContinuous.div_const [UniformSpace β] {f : β → α} (hf : UniformContinuous f)
    (a : α) : UniformContinuous fun x ↦ f x / a :=
  hf.div uniformContinuous_const


@[to_additive]
theorem uniformContinuous_div_const (a : α) : UniformContinuous fun b : α => b / a :=
  uniformContinuous_id.div_const _


@[to_additive UniformContinuous.const_nsmul]
theorem UniformContinuous.pow_const [UniformSpace β] {f : β → α} (hf : UniformContinuous f) :
    ∀ n : ℕ, UniformContinuous fun x => f x ^ n
  | 0 => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      ⊢ UniformContinuous fun x => HPow.hPow (f x) 0
    -/
    simp_rw [pow_zero]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      ⊢ UniformContinuous fun x => 1
    -/
    exact uniformContinuous_const
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    -/
    simp_rw [pow_succ']
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => HMul.hMul (f x) (HPow.hPow (f x) n)
    -/
    exact hf.mul (hf.pow_const n)
    /-
      🎉 no goals
    -/


@[to_additive uniformContinuous_const_nsmul]
theorem uniformContinuous_pow_const (n : ℕ) : UniformContinuous fun x : α => x ^ n :=
  uniformContinuous_id.pow_const n


@[to_additive UniformContinuous.const_zsmul]
theorem UniformContinuous.zpow_const [UniformSpace β] {f : β → α} (hf : UniformContinuous f) :
    ∀ n : ℤ, UniformContinuous fun x => f x ^ n
  | (n : ℕ) => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => HPow.hPow (f x) ↑n
    -/
    simp_rw [zpow_natCast]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => HPow.hPow (f x) n
    -/
    exact hf.pow_const _
    /-
      🎉 no goals
    -/
  | Int.negSucc n => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => HPow.hPow (f x) (Int.negSucc n)
    -/
    simp_rw [zpow_negSucc]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : UniformSpace α
      inst✝² : Group α
      inst✝¹ : UniformGroup α
      inst✝ : UniformSpace β
      f : β → α
      hf : UniformContinuous f
      n : Nat
      ⊢ UniformContinuous fun x => Inv.inv (HPow.hPow (f x) (HAdd.hAdd n 1))
    -/
    exact (hf.pow_const _).inv
    /-
      🎉 no goals
    -/


@[to_additive uniformContinuous_const_zsmul]
theorem uniformContinuous_zpow_const (n : ℤ) : UniformContinuous fun x : α => x ^ n :=
  uniformContinuous_id.zpow_const n


@[to_additive]
instance (priority := 10) UniformGroup.to_topologicalGroup : TopologicalGroup α where
  continuous_mul := uniformContinuous_mul.continuous
  continuous_inv := uniformContinuous_inv.continuous


@[to_additive]
instance [UniformSpace β] [Group β] [UniformGroup β] : UniformGroup (α × β) :=
  ⟨((uniformContinuous_fst.comp uniformContinuous_fst).div
          (uniformContinuous_fst.comp uniformContinuous_snd)).prod_mk
      ((uniformContinuous_snd.comp uniformContinuous_fst).div
        (uniformContinuous_snd.comp uniformContinuous_snd))⟩


@[to_additive]
theorem uniformity_translate_mul (a : α) : ((𝓤 α).map fun x : α × α => (x.1 * a, x.2 * a)) = 𝓤 α :=
  le_antisymm (uniformContinuous_id.mul uniformContinuous_const)
    (calc
      𝓤 α =
          ((𝓤 α).map fun x : α × α => (x.1 * a⁻¹, x.2 * a⁻¹)).map fun x : α × α =>
                                     /-
                                       α : Type u_1
                                       inst✝² : UniformSpace α
                                       inst✝¹ : Group α
                                       inst✝ : UniformGroup α
                                       a : α
                                       ⊢ Eq (uniformity α) (Filter.map (fun x => { fst := HMul.hMul x.1 a, snd := HMu …
                                     -/
            (x.1 * a, x.2 * a) := by simp [Filter.map_map, Function.comp_def]
                                     /-
                                       🎉 no goals
                                     -/
      _ ≤ (𝓤 α).map fun x : α × α => (x.1 * a, x.2 * a) :=
        Filter.map_mono (uniformContinuous_id.mul uniformContinuous_const)
      )


@[to_additive]
instance : UniformGroup αᵐᵒᵖ :=
  ⟨uniformContinuous_op.comp
      ((uniformContinuous_unop.comp uniformContinuous_snd).inv.mul <|
        uniformContinuous_unop.comp uniformContinuous_fst)⟩


@[to_additive]
theorem uniformGroup_sInf {us : Set (UniformSpace β)} (h : ∀ u ∈ us, @UniformGroup β u _) :
    @UniformGroup β (sInf us) _ :=
  -- Porting note: {_} does not find `sInf us` instance, see `continuousSMul_sInf`
  @UniformGroup.mk β (_) _ <|
    uniformContinuous_sInf_rng.mpr fun u hu =>
      uniformContinuous_sInf_dom₂ hu hu (@UniformGroup.uniformContinuous_div β u _ (h u hu))


@[to_additive]
theorem uniformGroup_iInf {ι : Sort*} {us' : ι → UniformSpace β}
    (h' : ∀ i, @UniformGroup β (us' i) _) : @UniformGroup β (⨅ i, us' i) _ := by
  /-
    β : Type u_2
    inst✝ : Group β
    ι : Sort u_3
    us' : ι → UniformSpace β
    h' : ∀ (i : ι), UniformGroup β
    ⊢ UniformGroup β
  -/
  rw [← sInf_range]
  /-
    β : Type u_2
    inst✝ : Group β
    ι : Sort u_3
    us' : ι → UniformSpace β
    h' : ∀ (i : ι), UniformGroup β
    ⊢ UniformGroup β
  -/
  exact uniformGroup_sInf (Set.forall_mem_range.mpr h')
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformGroup_inf {u₁ u₂ : UniformSpace β} (h₁ : @UniformGroup β u₁ _)
    (h₂ : @UniformGroup β u₂ _) : @UniformGroup β (u₁ ⊓ u₂) _ := by
  /-
    β : Type u_2
    inst✝ : Group β
    u₁ u₂ : UniformSpace β
    h₁ : UniformGroup β
    h₂ : UniformGroup β
    ⊢ UniformGroup β
  -/
  rw [inf_eq_iInf]
  /-
    β : Type u_2
    inst✝ : Group β
    u₁ u₂ : UniformSpace β
    h₁ : UniformGroup β
    h₂ : UniformGroup β
    ⊢ UniformGroup β
  -/
  refine uniformGroup_iInf fun b => ?_
  /-
    β : Type u_2
    inst✝ : Group β
    u₁ u₂ : UniformSpace β
    h₁ : UniformGroup β
    h₂ : UniformGroup β
    b : Bool
    ⊢ UniformGroup β
  -/
              /-
                🎉 no goals
              -/
  cases b <;> assumption
              /-
                🎉 no goals
              -/


@[to_additive]
theorem uniformity_eq_comap_nhds_one : 𝓤 α = comap (fun x : α × α => x.2 / x.1) (𝓝 (1 : α)) := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (uniformity α) (Filter.comap (fun x => HDiv.hDiv x.2 x.1) (nhds 1))
  -/
  rw [nhds_eq_comap_uniformity, Filter.comap_comap]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (uniformity α) (Filter.comap (Function.comp (Prod.mk 1) fun x => HDiv.hDi …
  -/
  refine le_antisymm (Filter.map_le_iff_le_comap.1 ?_) ?_
    /-
      case refine_1
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      ⊢ LE.le (Filter.map (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x.2 x.1) (un …
    -/
  · intro s hs
    /-
      case refine_1
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      ⊢ Membership.mem (Filter.map (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x.2 …
    -/
    rcases mem_uniformity_of_uniformContinuous_invariant uniformContinuous_div hs with ⟨t, ht, hts⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      ⊢ Membership.mem (Filter.map (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x.2 …
    -/
    refine mem_map.2 (mem_of_superset ht ?_)
    /-
      case refine_1.intro.intro
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      ⊢ HasSubset.Subset t (Set.preimage (Function.comp (Prod.mk 1) fun x => HDiv.hD …
    -/
    rintro ⟨a, b⟩
    /-
      case refine_1.intro.intro.mk
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      a b : α
      ⊢ Membership.mem t { fst := a, snd := b } → Membership.mem (Set.preimage (Func …
    -/
    simpa [subset_def] using hts a b a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      ⊢ LE.le (Filter.comap (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x.2 x.1) ( …
    -/
  · intro s hs
    /-
      case refine_2
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      ⊢ Membership.mem (Filter.comap (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x …
    -/
    rcases mem_uniformity_of_uniformContinuous_invariant uniformContinuous_mul hs with ⟨t, ht, hts⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      ⊢ Membership.mem (Filter.comap (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x …
    -/
    refine ⟨_, ht, ?_⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      ⊢ HasSubset.Subset (Set.preimage (Function.comp (Prod.mk 1) fun x => HDiv.hDiv …
    -/
    rintro ⟨a, b⟩
    /-
      case refine_2.intro.intro.mk
      α : Type u_1
      inst✝² : UniformSpace α
      inst✝¹ : Group α
      inst✝ : UniformGroup α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      hts : ∀ (a b c : α), Membership.mem t { fst := a, snd := b } → Membership.mem  …
      a b : α
      ⊢ Membership.mem (Set.preimage (Function.comp (Prod.mk 1) fun x => HDiv.hDiv x …
    -/
    simpa [subset_def] using hts 1 (b / a) a
    /-
      🎉 no goals
    -/


@[to_additive]
theorem uniformity_eq_comap_nhds_one_swapped :
    𝓤 α = comap (fun x : α × α => x.1 / x.2) (𝓝 (1 : α)) := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (uniformity α) (Filter.comap (fun x => HDiv.hDiv x.1 x.2) (nhds 1))
  -/
  rw [← comap_swap_uniformity, uniformity_eq_comap_nhds_one, comap_comap]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (Filter.comap (Function.comp (fun x => HDiv.hDiv x.2 x.1) Prod.swap) (nhd …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem UniformGroup.ext {G : Type*} [Group G] {u v : UniformSpace G} (hu : @UniformGroup G u _)
    (hv : @UniformGroup G v _)
    (h : @nhds _ u.toTopologicalSpace 1 = @nhds _ v.toTopologicalSpace 1) : u = v :=
  UniformSpace.ext <| by
    /-
      G : Type u_3
      inst✝ : Group G
      u v : UniformSpace G
      hu : UniformGroup G
      hv : UniformGroup G
      h : Eq (nhds 1) (nhds 1)
      ⊢ Eq (uniformity G) (uniformity G)
    -/
    rw [@uniformity_eq_comap_nhds_one _ u _ hu, @uniformity_eq_comap_nhds_one _ v _ hv, h]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem UniformGroup.ext_iff {G : Type*} [Group G] {u v : UniformSpace G}
    (hu : @UniformGroup G u _) (hv : @UniformGroup G v _) :
    u = v ↔ @nhds _ u.toTopologicalSpace 1 = @nhds _ v.toTopologicalSpace 1 :=
  ⟨fun h => h ▸ rfl, hu.ext hv⟩


@[to_additive]
theorem UniformGroup.uniformity_countably_generated [(𝓝 (1 : α)).IsCountablyGenerated] :
    (𝓤 α).IsCountablyGenerated := by
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : (nhds 1).IsCountablyGenerated
    ⊢ (uniformity α).IsCountablyGenerated
  -/
  rw [uniformity_eq_comap_nhds_one]
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    inst✝² : Group α
    inst✝¹ : UniformGroup α
    inst✝ : (nhds 1).IsCountablyGenerated
    ⊢ (Filter.comap (fun x => HDiv.hDiv x.2 x.1) (nhds 1)).IsCountablyGenerated
  -/
  exact Filter.comap.isCountablyGenerated _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformity_eq_comap_inv_mul_nhds_one :
    𝓤 α = comap (fun x : α × α => x.1⁻¹ * x.2) (𝓝 (1 : α)) := by
  rw [← comap_uniformity_mulOpposite, uniformity_eq_comap_nhds_one, ← op_one, ← comap_unop_nhds,
    comap_comap, comap_comap]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (Filter.comap (Function.comp MulOpposite.unop (Function.comp (fun x => HD …
  -/
  simp [Function.comp_def]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformity_eq_comap_inv_mul_nhds_one_swapped :
    𝓤 α = comap (fun x : α × α => x.2⁻¹ * x.1) (𝓝 (1 : α)) := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (uniformity α) (Filter.comap (fun x => HMul.hMul (Inv.inv x.2) x.1) (nhds …
  -/
  rw [← comap_swap_uniformity, uniformity_eq_comap_inv_mul_nhds_one, comap_comap]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ⊢ Eq (Filter.comap (Function.comp (fun x => HMul.hMul (Inv.inv x.1) x.2) Prod. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.HasBasis.uniformity_of_nhds_one {ι} {p : ι → Prop} {U : ι → Set α}
    (h : (𝓝 (1 : α)).HasBasis p U) :
    (𝓤 α).HasBasis p fun i => { x : α × α | x.2 / x.1 ∈ U i } := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (uniformity α).HasBasis p fun i => setOf fun x => Membership.mem (U i) (HDiv …
  -/
  rw [uniformity_eq_comap_nhds_one]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (Filter.comap (fun x => HDiv.hDiv x.2 x.1) (nhds 1)).HasBasis p fun i => set …
  -/
  exact h.comap _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.HasBasis.uniformity_of_nhds_one_inv_mul {ι} {p : ι → Prop} {U : ι → Set α}
    (h : (𝓝 (1 : α)).HasBasis p U) :
    (𝓤 α).HasBasis p fun i => { x : α × α | x.1⁻¹ * x.2 ∈ U i } := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (uniformity α).HasBasis p fun i => setOf fun x => Membership.mem (U i) (HMul …
  -/
  rw [uniformity_eq_comap_inv_mul_nhds_one]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (Filter.comap (fun x => HMul.hMul (Inv.inv x.1) x.2) (nhds 1)).HasBasis p fu …
  -/
  exact h.comap _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.HasBasis.uniformity_of_nhds_one_swapped {ι} {p : ι → Prop} {U : ι → Set α}
    (h : (𝓝 (1 : α)).HasBasis p U) :
    (𝓤 α).HasBasis p fun i => { x : α × α | x.1 / x.2 ∈ U i } := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (uniformity α).HasBasis p fun i => setOf fun x => Membership.mem (U i) (HDiv …
  -/
  rw [uniformity_eq_comap_nhds_one_swapped]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (Filter.comap (fun x => HDiv.hDiv x.1 x.2) (nhds 1)).HasBasis p fun i => set …
  -/
  exact h.comap _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.HasBasis.uniformity_of_nhds_one_inv_mul_swapped {ι} {p : ι → Prop} {U : ι → Set α}
    (h : (𝓝 (1 : α)).HasBasis p U) :
    (𝓤 α).HasBasis p fun i => { x : α × α | x.2⁻¹ * x.1 ∈ U i } := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (uniformity α).HasBasis p fun i => setOf fun x => Membership.mem (U i) (HMul …
  -/
  rw [uniformity_eq_comap_inv_mul_nhds_one_swapped]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    inst✝¹ : Group α
    inst✝ : UniformGroup α
    ι : Sort u_3
    p : ι → Prop
    U : ι → Set α
    h : (nhds 1).HasBasis p U
    ⊢ (Filter.comap (fun x => HMul.hMul (Inv.inv x.2) x.1) (nhds 1)).HasBasis p fu …
  -/
  exact h.comap _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformContinuous_of_tendsto_one {hom : Type*} [UniformSpace β] [Group β] [UniformGroup β]
    [FunLike hom α β] [MonoidHomClass hom α β] {f : hom} (h : Tendsto f (𝓝 1) (𝓝 1)) :
    UniformContinuous f := by
  have :
    ((fun x : β × β => x.2 / x.1) ∘ fun x : α × α => (f x.1, f x.2)) = fun x : α × α =>
      f (x.2 / x.1) := by ext; simp only [Function.comp_apply, map_div]
  rw [UniformContinuous, uniformity_eq_comap_nhds_one α, uniformity_eq_comap_nhds_one β,
    tendsto_comap_iff, this]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : UniformSpace α
    inst✝⁶ : Group α
    inst✝⁵ : UniformGroup α
    hom : Type u_3
    inst✝⁴ : UniformSpace β
    inst✝³ : Group β
    inst✝² : UniformGroup β
    inst✝¹ : FunLike hom α β
    inst✝ : MonoidHomClass hom α β
    f : hom
    h : Filter.Tendsto (⇑f) (nhds 1) (nhds 1)
    this : Eq (Function.comp (fun x => HDiv.hDiv x.2 x.1) fun x => { fst := f x.1, …
    ⊢ Filter.Tendsto (fun x => f (HDiv.hDiv x.2 x.1)) (Filter.comap (fun x => HDiv …
  -/
  exact Tendsto.comp h tendsto_comap
  /-
    🎉 no goals
  -/


/-- A group homomorphism (a bundled morphism of a type that implements `MonoidHomClass`) between
two uniform groups is uniformly continuous provided that it is continuous at one. See also
`continuous_of_continuousAt_one`. -/
@[to_additive "An additive group homomorphism (a bundled morphism of a type that implements
`AddMonoidHomClass`) between two uniform additive groups is uniformly continuous provided that it
is continuous at zero. See also `continuous_of_continuousAt_zero`."]
theorem uniformContinuous_of_continuousAt_one {hom : Type*} [UniformSpace β] [Group β]
    [UniformGroup β] [FunLike hom α β] [MonoidHomClass hom α β]
    (f : hom) (hf : ContinuousAt f 1) :
    UniformContinuous f :=
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         inst✝⁷ : UniformSpace α
                                         inst✝⁶ : Group α
                                         inst✝⁵ : UniformGroup α
                                         hom : Type u_3
                                         inst✝⁴ : UniformSpace β
                                         inst✝³ : Group β
                                         inst✝² : UniformGroup β
                                         inst✝¹ : FunLike hom α β
                                         inst✝ : MonoidHomClass hom α β
                                         f : hom
                                         hf : ContinuousAt (⇑f) 1
                                         ⊢ Filter.Tendsto (⇑f) (nhds 1) (nhds 1)
                                       -/
  uniformContinuous_of_tendsto_one (by simpa using hf.tendsto)
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive]
theorem MonoidHom.uniformContinuous_of_continuousAt_one [UniformSpace β] [Group β] [UniformGroup β]
    (f : α →* β) (hf : ContinuousAt f 1) : UniformContinuous f :=
  _root_.uniformContinuous_of_continuousAt_one f hf


/-- A homomorphism from a uniform group to a discrete uniform group is continuous if and only if
its kernel is open. -/
@[to_additive "A homomorphism from a uniform additive group to a discrete uniform additive group is
continuous if and only if its kernel is open."]
theorem UniformGroup.uniformContinuous_iff_isOpen_ker {hom : Type*} [UniformSpace β]
    [DiscreteTopology β] [Group β] [UniformGroup β] [FunLike hom α β] [MonoidHomClass hom α β]
    {f : hom} :
    UniformContinuous f ↔ IsOpen ((f : α →* β).ker : Set α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁸ : UniformSpace α
    inst✝⁷ : Group α
    inst✝⁶ : UniformGroup α
    hom : Type u_3
    inst✝⁵ : UniformSpace β
    inst✝⁴ : DiscreteTopology β
    inst✝³ : Group β
    inst✝² : UniformGroup β
    inst✝¹ : FunLike hom α β
    inst✝ : MonoidHomClass hom α β
    f : hom
    ⊢ Iff (UniformContinuous ⇑f) (IsOpen ↑(↑f).ker)
  -/
  refine ⟨fun hf => ?_, fun hf => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁸ : UniformSpace α
      inst✝⁷ : Group α
      inst✝⁶ : UniformGroup α
      hom : Type u_3
      inst✝⁵ : UniformSpace β
      inst✝⁴ : DiscreteTopology β
      inst✝³ : Group β
      inst✝² : UniformGroup β
      inst✝¹ : FunLike hom α β
      inst✝ : MonoidHomClass hom α β
      f : hom
      hf : UniformContinuous ⇑f
      ⊢ IsOpen ↑(↑f).ker
    -/
  · apply (isOpen_discrete ({1} : Set β)).preimage hf.continuous
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁸ : UniformSpace α
      inst✝⁷ : Group α
      inst✝⁶ : UniformGroup α
      hom : Type u_3
      inst✝⁵ : UniformSpace β
      inst✝⁴ : DiscreteTopology β
      inst✝³ : Group β
      inst✝² : UniformGroup β
      inst✝¹ : FunLike hom α β
      inst✝ : MonoidHomClass hom α β
      f : hom
      hf : IsOpen ↑(↑f).ker
      ⊢ UniformContinuous ⇑f
    -/
  · apply uniformContinuous_of_continuousAt_one
    /-
      case refine_2.hf
      α : Type u_1
      β : Type u_2
      inst✝⁸ : UniformSpace α
      inst✝⁷ : Group α
      inst✝⁶ : UniformGroup α
      hom : Type u_3
      inst✝⁵ : UniformSpace β
      inst✝⁴ : DiscreteTopology β
      inst✝³ : Group β
      inst✝² : UniformGroup β
      inst✝¹ : FunLike hom α β
      inst✝ : MonoidHomClass hom α β
      f : hom
      hf : IsOpen ↑(↑f).ker
      ⊢ ContinuousAt (⇑f) 1
    -/
    rw [ContinuousAt, nhds_discrete β, map_one, tendsto_pure]
    /-
      case refine_2.hf
      α : Type u_1
      β : Type u_2
      inst✝⁸ : UniformSpace α
      inst✝⁷ : Group α
      inst✝⁶ : UniformGroup α
      hom : Type u_3
      inst✝⁵ : UniformSpace β
      inst✝⁴ : DiscreteTopology β
      inst✝³ : Group β
      inst✝² : UniformGroup β
      inst✝¹ : FunLike hom α β
      inst✝ : MonoidHomClass hom α β
      f : hom
      hf : IsOpen ↑(↑f).ker
      ⊢ Filter.Eventually (fun x => Eq (f x) 1) (nhds 1)
    -/
    exact hf.mem_nhds (map_one f)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-18")] alias UniformGroup.uniformContinuous_iff_open_ker :=
  UniformGroup.uniformContinuous_iff_isOpen_ker

@[deprecated (since := "2024-11-18")] alias UniformAddGroup.uniformContinuous_iff_open_ker :=
  UniformAddGroup.uniformContinuous_iff_isOpen_ker


@[to_additive]
theorem uniformContinuous_monoidHom_of_continuous {hom : Type*} [UniformSpace β] [Group β]
    [UniformGroup β] [FunLike hom α β] [MonoidHomClass hom α β] {f : hom} (h : Continuous f) :
    UniformContinuous f :=
  uniformContinuous_of_tendsto_one <|
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            inst✝⁷ : UniformSpace α
                                            inst✝⁶ : Group α
                                            inst✝⁵ : UniformGroup α
                                            hom : Type u_3
                                            inst✝⁴ : UniformSpace β
                                            inst✝³ : Group β
                                            inst✝² : UniformGroup β
                                            inst✝¹ : FunLike hom α β
                                            inst✝ : MonoidHomClass hom α β
                                            f : hom
                                            h : Continuous ⇑f
                                            this : Filter.Tendsto (⇑f) (nhds 1) (nhds (f 1))
                                            ⊢ Filter.Tendsto (⇑f) (nhds 1) (nhds 1)
                                          -/
    suffices Tendsto f (𝓝 1) (𝓝 (f 1)) by rwa [map_one] at this
                                          /-
                                            🎉 no goals
                                          -/
    h.tendsto 1


/-- The right uniformity on a topological group (as opposed to the left uniformity).

Warning: in general the right and left uniformities do not coincide and so one does not obtain a
`UniformGroup` structure. Two important special cases where they _do_ coincide are for
commutative groups (see `comm_topologicalGroup_is_uniform`) and for compact groups (see
`topologicalGroup_is_uniform_of_compactSpace`). -/
@[to_additive "The right uniformity on a topological additive group (as opposed to the left
uniformity).

Warning: in general the right and left uniformities do not coincide and so one does not obtain a
`UniformAddGroup` structure. Two important special cases where they _do_ coincide are for
commutative additive groups (see `comm_topologicalAddGroup_is_uniform`) and for compact
additive groups (see `topologicalAddGroup_is_uniform_of_compactSpace`)."]
def TopologicalGroup.toUniformSpace : UniformSpace G where
  uniformity := comap (fun p : G × G => p.2 / p.1) (𝓝 1)
  symm :=
    have : Tendsto (fun p : G × G ↦ (p.2 / p.1)⁻¹) (comap (fun p : G × G ↦ p.2 / p.1) (𝓝 1))
      (𝓝 1⁻¹) := tendsto_id.inv.comp tendsto_comap
       /-
         G : Type u_1
         inst✝² : Group G
         inst✝¹ : TopologicalSpace G
         inst✝ : TopologicalGroup G
         this : Filter.Tendsto (fun p => Inv.inv (HDiv.hDiv p.2 p.1)) (Filter.comap (fu …
         ⊢ Filter.Tendsto Prod.swap (Filter.comap (fun p => HDiv.hDiv p.2 p.1) (nhds 1) …
       -/
    by simpa [tendsto_comap_iff]
       /-
         🎉 no goals
       -/
  comp := Tendsto.le_comap fun U H ↦ by
    /-
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      U : Set G
      H : Membership.mem (nhds 1) U
      ⊢ Membership.mem (Filter.map (fun p => HDiv.hDiv p.2 p.1) ((Filter.comap (fun  …
    -/
    rcases exists_nhds_one_split H with ⟨V, V_nhds, V_mul⟩
    /-
      case intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      U : Set G
      H : Membership.mem (nhds 1) U
      V : Set G
      V_nhds : Membership.mem (nhds 1) V
      V_mul : ∀ (v : G), Membership.mem V v → ∀ (w : G), Membership.mem V w → Member …
      ⊢ Membership.mem (Filter.map (fun p => HDiv.hDiv p.2 p.1) ((Filter.comap (fun  …
    -/
    refine mem_map.2 (mem_of_superset (mem_lift' <| preimage_mem_comap V_nhds) ?_)
    /-
      case intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      U : Set G
      H : Membership.mem (nhds 1) U
      V : Set G
      V_nhds : Membership.mem (nhds 1) V
      V_mul : ∀ (v : G), Membership.mem V v → ∀ (w : G), Membership.mem V w → Member …
      ⊢ HasSubset.Subset (compRel (Set.preimage (fun p => HDiv.hDiv p.2 p.1) V) (Set …
    -/
    rintro ⟨x, y⟩ ⟨z, hz₁, hz₂⟩
    /-
      case intro.intro.mk.intro.intro
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalGroup G
      U : Set G
      H : Membership.mem (nhds 1) U
      V : Set G
      V_nhds : Membership.mem (nhds 1) V
      V_mul : ∀ (v : G), Membership.mem V v → ∀ (w : G), Membership.mem V w → Member …
      x y z : G
      hz₁ : Membership.mem (Set.preimage (fun p => HDiv.hDiv p.2 p.1) V) { fst := {  …
      hz₂ : Membership.mem (Set.preimage (fun p => HDiv.hDiv p.2 p.1) V) { fst := z, …
      ⊢ Membership.mem (Set.preimage (fun p => HDiv.hDiv p.2 p.1) U) { fst := x, snd …
    -/
    simpa using V_mul _ hz₂ _ hz₁
    /-
      🎉 no goals
    -/
                                   /-
                                     G : Type u_1
                                     inst✝² : Group G
                                     inst✝¹ : TopologicalSpace G
                                     inst✝ : TopologicalGroup G
                                     x✝ : G
                                     ⊢ Eq (nhds x✝) (Filter.comap (Prod.mk x✝) (Filter.comap (fun p => HDiv.hDiv p. …
                                   -/
  nhds_eq_comap_uniformity _ := by simp only [comap_comap, Function.comp_def, nhds_translation_div]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem uniformity_eq_comap_nhds_one' : 𝓤 G = comap (fun p : G × G => p.2 / p.1) (𝓝 (1 : G)) :=
  rfl


@[to_additive]
-- Porting note: renamed theorem to conform to naming convention
theorem comm_topologicalGroup_is_uniform : UniformGroup G := by
  /-
    G : Type u_1
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    ⊢ UniformGroup G
  -/
  constructor
  simp only [UniformContinuous, uniformity_prod_eq_prod, uniformity_eq_comap_nhds_one',
    tendsto_comap_iff, tendsto_map'_iff, prod_comap_comap_eq, Function.comp_def,
    div_div_div_comm _ (Prod.snd (Prod.snd _)), ← nhds_prod_eq, Prod.mk_one_one]
  /-
    case uniformContinuous_div
    G : Type u_1
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HDiv.hDiv x.1.2 x.1.1) (HDiv.hDiv x.2.2  …
  -/
  exact (continuous_div'.tendsto' 1 1 (div_one 1)).comp tendsto_comap
  /-
    🎉 no goals
  -/


@[to_additive]
theorem UniformGroup.toUniformSpace_eq {G : Type*} [u : UniformSpace G] [Group G]
    [UniformGroup G] : TopologicalGroup.toUniformSpace G = u := by
  /-
    G : Type u_2
    u : UniformSpace G
    inst✝¹ : Group G
    inst✝ : UniformGroup G
    ⊢ Eq (TopologicalGroup.toUniformSpace G) u
  -/
  ext : 1
  /-
    case h
    G : Type u_2
    u : UniformSpace G
    inst✝¹ : Group G
    inst✝ : UniformGroup G
    ⊢ Eq (uniformity G) (uniformity G)
  -/
  rw [uniformity_eq_comap_nhds_one' G, uniformity_eq_comap_nhds_one G]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tendsto_div_comap_self (de : IsDenseInducing e) (x₀ : α) :
    Tendsto (fun t : β × β => t.2 / t.1) ((comap fun p : β × β => (e p.1, e p.2)) <| 𝓝 (x₀, x₀))
      (𝓝 1) := by
  have comm : ((fun x : α × α => x.2 / x.1) ∘ fun t : β × β => (e t.1, e t.2)) =
      e ∘ fun t : β × β => t.2 / t.1 := by
    ext t
    change e t.2 / e t.1 = e (t.2 / t.1)
    rw [← map_div e t.2 t.1]
  have lim : Tendsto (fun x : α × α => x.2 / x.1) (𝓝 (x₀, x₀)) (𝓝 (e 1)) := by
    simpa using (continuous_div'.comp (@continuous_swap α α _ _)).tendsto (x₀, x₀)
  /-
    α : Type u_1
    β : Type u_2
    hom : Type u_3
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : Group α
    inst✝⁴ : TopologicalGroup α
    inst✝³ : TopologicalSpace β
    inst✝² : Group β
    inst✝¹ : FunLike hom β α
    inst✝ : MonoidHomClass hom β α
    e : hom
    de : IsDenseInducing ⇑e
    x₀ : α
    comm : Eq (Function.comp (fun x => HDiv.hDiv x.2 x.1) fun t => { fst := e t.1, …
    lim : Filter.Tendsto (fun x => HDiv.hDiv x.2 x.1) (nhds { fst := x₀, snd := x₀ …
    ⊢ Filter.Tendsto (fun t => HDiv.hDiv t.2 t.1) (Filter.comap (fun p => { fst := …
  -/
  simpa using de.tendsto_comap_nhds_nhds lim comm
  /-
    🎉 no goals
  -/


include W'_nhd in
private theorem extend_Z_bilin_aux (x₀ : α) (y₁ : δ) : ∃ U₂ ∈ comap e (𝓝 x₀), ∀ x ∈ U₂, ∀ x' ∈ U₂,
    (fun p : β × δ => φ p.1 p.2) (x' - x, y₁) ∈ W' := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  let Nx := 𝓝 x₀
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  let ee := fun u : β × β => (e u.1, e u.2)
  have lim1 : Tendsto (fun a : β × β => (a.2 - a.1, y₁))
      (comap e Nx ×ˢ comap e Nx) (𝓝 (0, y₁)) := by
    have := Tendsto.prod_mk (tendsto_sub_comap_self de x₀)
      (tendsto_const_nhds : Tendsto (fun _ : β × β => y₁) (comap ee <| 𝓝 (x₀, x₀)) (𝓝 y₁))
    rw [nhds_prod_eq, prod_comap_comap_eq, ← nhds_prod_eq]
    exact (this : _)
  have lim2 : Tendsto (fun p : β × δ => φ p.1 p.2) (𝓝 (0, y₁)) (𝓝 0) := by
    simpa using hφ.tendsto (0, y₁)
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  have lim := lim2.comp lim1
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : Filter.Tendsto (Function.comp (fun p => (φ p.1) p.2) fun a => { fst := H …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  rw [tendsto_prod_self_iff] at lim
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : ∀ (W : Set G), Membership.mem (nhds 0) W → Exists fun U => And (Membersh …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (x …
  -/
  simp_rw [forall_mem_comm]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    G : Type u_5
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : AddCommGroup α
    inst✝⁶ : TopologicalAddGroup α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : AddCommGroup β
    inst✝³ : TopologicalSpace δ
    inst✝² : AddCommGroup δ
    inst✝¹ : UniformSpace G
    inst✝ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    x₀ : α
    y₁ : δ
    Nx : Filter α := nhds x₀
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    lim1 : Filter.Tendsto (fun a => { fst := HSub.hSub a.2 a.1, snd := y₁ }) (SPro …
    lim2 : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := y₁ }) (n …
    lim : ∀ (W : Set G), Membership.mem (nhds 0) W → Exists fun U => And (Membersh …
    ⊢ Exists fun U₂ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂) (∀ (a …
  -/
  exact lim W' W'_nhd
  /-
    🎉 no goals
  -/


include df W'_nhd in
private theorem extend_Z_bilin_key (x₀ : α) (y₀ : γ) : ∃ U ∈ comap e (𝓝 x₀), ∃ V ∈ comap f (𝓝 y₀),
    ∀ x ∈ U, ∀ x' ∈ U, ∀ (y) (_ : y ∈ V) (y') (_ : y' ∈ V),
    (fun p : β × δ => φ p.1 p.2) (x', y') - (fun p : β × δ => φ p.1 p.2) (x, y) ∈ W' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  let ee := fun u : β × β => (e u.1, e u.2)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  let ff := fun u : δ × δ => (f u.1, f u.2)
  have lim_φ : Filter.Tendsto (fun p : β × δ => φ p.1 p.2) (𝓝 (0, 0)) (𝓝 0) := by
    simpa using hφ.tendsto (0, 0)
  have lim_φ_sub_sub :
    Tendsto (fun p : (β × β) × δ × δ => (fun p : β × δ => φ p.1 p.2) (p.1.2 - p.1.1, p.2.2 - p.2.1))
      ((comap ee <| 𝓝 (x₀, x₀)) ×ˢ (comap ff <| 𝓝 (y₀, y₀))) (𝓝 0) := by
    have lim_sub_sub :
      Tendsto (fun p : (β × β) × δ × δ => (p.1.2 - p.1.1, p.2.2 - p.2.1))
        (comap ee (𝓝 (x₀, x₀)) ×ˢ comap ff (𝓝 (y₀, y₀))) (𝓝 0 ×ˢ 𝓝 0) := by
      have := Filter.prod_mono (tendsto_sub_comap_self de x₀) (tendsto_sub_comap_self df y₀)
      rwa [prod_map_map_eq] at this
    rw [← nhds_prod_eq] at lim_sub_sub
    exact Tendsto.comp lim_φ lim_sub_sub
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases exists_nhds_zero_quarter W'_nhd with ⟨W, W_nhd, W4⟩
  have :
    ∃ U₁ ∈ comap e (𝓝 x₀), ∃ V₁ ∈ comap f (𝓝 y₀), ∀ (x) (_ : x ∈ U₁) (x') (_ : x' ∈ U₁),
      ∀ (y) (_ : y ∈ V₁) (y') (_ : y' ∈ V₁), (fun p : β × δ => φ p.1 p.2) (x' - x, y' - y) ∈ W := by
    rcases tendsto_prod_iff.1 lim_φ_sub_sub W W_nhd with ⟨U, U_in, V, V_in, H⟩
    rw [nhds_prod_eq, ← prod_comap_comap_eq, mem_prod_same_iff] at U_in V_in
    rcases U_in with ⟨U₁, U₁_in, HU₁⟩
    rcases V_in with ⟨V₁, V₁_in, HV₁⟩
    exists U₁, U₁_in, V₁, V₁_in
    intro x x_in x' x'_in y y_in y' y'_in
    exact H _ _ (HU₁ (mk_mem_prod x_in x'_in)) (HV₁ (mk_mem_prod y_in y'_in))
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    this : Exists fun U₁ => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁)  …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases this with ⟨U₁, U₁_nhd, V₁, V₁_nhd, H⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  obtain ⟨x₁, x₁_in⟩ : U₁.Nonempty := (de.comap_nhds_neBot _).nonempty_of_mem U₁_nhd
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  obtain ⟨y₁, y₁_in⟩ : V₁.Nonempty := (df.comap_nhds_neBot _).nonempty_of_mem V₁_nhd
  have cont_flip : Continuous fun p : δ × β => φ.flip p.1 p.2 := by
    show Continuous ((fun p : β × δ => φ p.1 p.2) ∘ Prod.swap)
    exact hφ.comp continuous_swap
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases extend_Z_bilin_aux de hφ W_nhd x₀ y₁ with ⟨U₂, U₂_nhd, HU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  rcases extend_Z_bilin_aux df cont_flip W_nhd y₀ x₁ with ⟨V₂, V₂_nhd, HV⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    ⊢ Exists fun U => And (Membership.mem (Filter.comap (⇑e) (nhds x₀)) U) (Exists …
  -/
  exists U₁ ∩ U₂, inter_mem U₁_nhd U₂_nhd, V₁ ∩ V₂, inter_mem V₁_nhd V₂_nhd
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    ⊢ ∀ (x : β), Membership.mem (Inter.inter U₁ U₂) x → ∀ (x' : β), Membership.mem …
  -/
  rintro x ⟨xU₁, xU₂⟩ x' ⟨x'U₁, x'U₂⟩ y ⟨yV₁, yV₂⟩ y' ⟨y'V₁, y'V₂⟩
  have key_formula : φ x' y' - φ x y
    = φ (x' - x) y₁ + φ (x' - x) (y' - y₁) + φ x₁ (y' - y) + φ (x - x₁) (y' - y) := by simp; abel
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    ⊢ Membership.mem W' (HSub.hSub ((fun p => (φ p.1) p.2) { fst := x', snd := y'  …
  -/
  rw [key_formula]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₁ := HU x xU₂ x' x'U₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₂ := H x xU₁ x' x'U₁ y₁ y₁_in y' y'V₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₃ := HV y yV₂ y' y'V₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₃ : Membership.mem W ((fun p => (φ.flip p.1) p.2) { fst := HSub.hSub y' y, sn …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  have h₄ := H x₁ x₁_in x xU₁ y yV₁ y' y'V₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    G : Type u_5
    inst✝¹² : TopologicalSpace α
    inst✝¹¹ : AddCommGroup α
    inst✝¹⁰ : TopologicalAddGroup α
    inst✝⁹ : TopologicalSpace β
    inst✝⁸ : AddCommGroup β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : AddCommGroup γ
    inst✝⁵ : TopologicalAddGroup γ
    inst✝⁴ : TopologicalSpace δ
    inst✝³ : AddCommGroup δ
    inst✝² : UniformSpace G
    inst✝¹ : AddCommGroup G
    e : AddMonoidHom β α
    de : IsDenseInducing ⇑e
    f : AddMonoidHom δ γ
    df : IsDenseInducing ⇑f
    φ : AddMonoidHom β (AddMonoidHom δ G)
    hφ : Continuous fun p => (φ p.1) p.2
    W' : Set G
    W'_nhd : Membership.mem (nhds 0) W'
    inst✝ : UniformAddGroup G
    x₀ : α
    y₀ : γ
    ee : Prod β β → Prod α α := fun u => { fst := e u.1, snd := e u.2 }
    ff : Prod δ δ → Prod γ γ := fun u => { fst := f u.1, snd := f u.2 }
    lim_φ : Filter.Tendsto (fun p => (φ p.1) p.2) (nhds { fst := 0, snd := 0 }) (n …
    lim_φ_sub_sub : Filter.Tendsto (fun p => (fun p => (φ p.1) p.2) { fst := HSub. …
    W : Set G
    W_nhd : Membership.mem (nhds 0) W
    W4 : ∀ {v w s t : G}, Membership.mem W v → Membership.mem W w → Membership.mem …
    U₁ : Set β
    U₁_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₁
    V₁ : Set δ
    V₁_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₁
    H : ∀ (x : β), Membership.mem U₁ x → ∀ (x' : β), Membership.mem U₁ x' → ∀ (y : …
    x₁ : β
    x₁_in : Membership.mem U₁ x₁
    y₁ : δ
    y₁_in : Membership.mem V₁ y₁
    cont_flip : Continuous fun p => (φ.flip p.1) p.2
    U₂ : Set β
    U₂_nhd : Membership.mem (Filter.comap (⇑e) (nhds x₀)) U₂
    HU : ∀ (x : β), Membership.mem U₂ x → ∀ (x' : β), Membership.mem U₂ x' → Membe …
    V₂ : Set δ
    V₂_nhd : Membership.mem (Filter.comap (⇑f) (nhds y₀)) V₂
    HV : ∀ (x : δ), Membership.mem V₂ x → ∀ (x' : δ), Membership.mem V₂ x' → Membe …
    x : β
    xU₁ : Membership.mem U₁ x
    xU₂ : Membership.mem U₂ x
    x' : β
    x'U₁ : Membership.mem U₁ x'
    x'U₂ : Membership.mem U₂ x'
    y : δ
    yV₁ : Membership.mem V₁ y
    yV₂ : Membership.mem V₂ y
    y' : δ
    y'V₁ : Membership.mem V₁ y'
    y'V₂ : Membership.mem V₂ y'
    key_formula : Eq (HSub.hSub ((φ x') y') ((φ x) y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
    h₁ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₂ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x' x, snd :=  …
    h₃ : Membership.mem W ((fun p => (φ.flip p.1) p.2) { fst := HSub.hSub y' y, sn …
    h₄ : Membership.mem W ((fun p => (φ p.1) p.2) { fst := HSub.hSub x x₁, snd :=  …
    ⊢ Membership.mem W' (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((φ (HSub.hSub x' x)) y₁) …
  -/
  exact W4 h₁ h₂ h₃ h₄
  /-
    🎉 no goals
  -/



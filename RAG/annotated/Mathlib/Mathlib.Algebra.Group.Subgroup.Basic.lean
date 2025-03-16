@[to_additive]
theorem div_mem_comm_iff {a b : G} : a / b ∈ H ↔ b / a ∈ H :=
  inv_div b a ▸ inv_mem_iff


@[to_additive]
protected theorem div_mem_comm_iff {a b : G} : a / b ∈ H ↔ b / a ∈ H :=
  div_mem_comm_iff


/-- Given `Subgroup`s `H`, `K` of groups `G`, `N` respectively, `H × K` as a subgroup of `G × N`. -/
@[to_additive prod
      "Given `AddSubgroup`s `H`, `K` of `AddGroup`s `A`, `B` respectively, `H × K`
      as an `AddSubgroup` of `A × B`."]
def prod (H : Subgroup G) (K : Subgroup N) : Subgroup (G × N) :=
  { Submonoid.prod H.toSubmonoid K.toSubmonoid with
    inv_mem' := fun hx => ⟨H.inv_mem' hx.1, K.inv_mem' hx.2⟩ }


@[to_additive coe_prod]
theorem coe_prod (H : Subgroup G) (K : Subgroup N) :
    (H.prod K : Set (G × N)) = (H : Set G) ×ˢ (K : Set N) :=
  rfl


@[to_additive mem_prod]
theorem mem_prod {H : Subgroup G} {K : Subgroup N} {p : G × N} : p ∈ H.prod K ↔ p.1 ∈ H ∧ p.2 ∈ K :=
  Iff.rfl


@[to_additive prod_mono]
theorem prod_mono : ((· ≤ ·) ⇒ (· ≤ ·) ⇒ (· ≤ ·)) (@prod G _ N _) (@prod G _ N _) :=
  fun _s _s' hs _t _t' ht => Set.prod_mono hs ht


@[to_additive prod_mono_right]
theorem prod_mono_right (K : Subgroup G) : Monotone fun t : Subgroup N => K.prod t :=
  prod_mono (le_refl K)


@[to_additive prod_mono_left]
theorem prod_mono_left (H : Subgroup N) : Monotone fun K : Subgroup G => K.prod H := fun _ _ hs =>
  prod_mono hs (le_refl H)


@[to_additive prod_top]
theorem prod_top (K : Subgroup G) : K.prod (⊤ : Subgroup N) = K.comap (MonoidHom.fst G N) :=
                  /-
                    G : Type u_1
                    inst✝¹ : Group G
                    N : Type u_5
                    inst✝ : Group N
                    K : Subgroup G
                    x : Prod G N
                    ⊢ Iff (Membership.mem (K.prod Top.top) x) (Membership.mem (Subgroup.comap (Mon …
                  -/
  ext fun x => by simp [mem_prod, MonoidHom.coe_fst]
                  /-
                    🎉 no goals
                  -/


@[to_additive top_prod]
theorem top_prod (H : Subgroup N) : (⊤ : Subgroup G).prod H = H.comap (MonoidHom.snd G N) :=
                  /-
                    G : Type u_1
                    inst✝¹ : Group G
                    N : Type u_5
                    inst✝ : Group N
                    H : Subgroup N
                    x : Prod G N
                    ⊢ Iff (Membership.mem (Top.top.prod H) x) (Membership.mem (Subgroup.comap (Mon …
                  -/
  ext fun x => by simp [mem_prod, MonoidHom.coe_snd]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp) top_prod_top]
theorem top_prod_top : (⊤ : Subgroup G).prod (⊤ : Subgroup N) = ⊤ :=
  (top_prod _).trans <| comap_top _


@[to_additive]
theorem bot_prod_bot : (⊥ : Subgroup G).prod (⊥ : Subgroup N) = ⊥ :=
                              /-
                                G : Type u_1
                                inst✝¹ : Group G
                                N : Type u_5
                                inst✝ : Group N
                                ⊢ Eq ↑(Bot.bot.prod Bot.bot) ↑Bot.bot
                              -/
  SetLike.coe_injective <| by simp [coe_prod]
                              /-
                                🎉 no goals
                              -/


@[to_additive le_prod_iff]
theorem le_prod_iff {H : Subgroup G} {K : Subgroup N} {J : Subgroup (G × N)} :
    J ≤ H.prod K ↔ map (MonoidHom.fst G N) J ≤ H ∧ map (MonoidHom.snd G N) J ≤ K := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    K : Subgroup N
    J : Subgroup (Prod G N)
    ⊢ Iff (LE.le J (H.prod K)) (And (LE.le (Subgroup.map (MonoidHom.fst G N) J) H) …
  -/
  simpa only [← Subgroup.toSubmonoid_le] using Submonoid.le_prod_iff
  /-
    🎉 no goals
  -/


@[to_additive prod_le_iff]
theorem prod_le_iff {H : Subgroup G} {K : Subgroup N} {J : Subgroup (G × N)} :
    H.prod K ≤ J ↔ map (MonoidHom.inl G N) H ≤ J ∧ map (MonoidHom.inr G N) K ≤ J := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    K : Subgroup N
    J : Subgroup (Prod G N)
    ⊢ Iff (LE.le (H.prod K) J) (And (LE.le (Subgroup.map (MonoidHom.inl G N) H) J) …
  -/
  simpa only [← Subgroup.toSubmonoid_le] using Submonoid.prod_le_iff
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) prod_eq_bot_iff]
theorem prod_eq_bot_iff {H : Subgroup G} {K : Subgroup N} : H.prod K = ⊥ ↔ H = ⊥ ∧ K = ⊥ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    K : Subgroup N
    ⊢ Iff (Eq (H.prod K) Bot.bot) (And (Eq H Bot.bot) (Eq K Bot.bot))
  -/
  simpa only [← Subgroup.toSubmonoid_inj] using Submonoid.prod_eq_bot_iff
  /-
    🎉 no goals
  -/


@[to_additive closure_prod]
theorem closure_prod {s : Set G} {t : Set N} (hs : 1 ∈ s) (ht : 1 ∈ t) :
    closure (s ×ˢ t) = (closure s).prod (closure t) :=
  le_antisymm
    (closure_le _ |>.2 <| Set.prod_subset_prod_iff.2 <| .inl ⟨subset_closure, subset_closure⟩)
    (prod_le_iff.2 ⟨
      map_le_iff_le_comap.2 <| closure_le _ |>.2 fun _x hx => subset_closure ⟨hx, ht⟩,
      map_le_iff_le_comap.2 <| closure_le _ |>.2 fun _y hy => subset_closure ⟨hs, hy⟩⟩)


/-- Product of subgroups is isomorphic to their product as groups. -/
@[to_additive prodEquiv
      "Product of additive subgroups is isomorphic to their product
      as additive groups"]
def prodEquiv (H : Subgroup G) (K : Subgroup N) : H.prod K ≃* H × K :=
  { Equiv.Set.prod (H : Set G) (K : Set N) with map_mul' := fun _ _ => rfl }


/-- A version of `Set.pi` for submonoids. Given an index set `I` and a family of submodules
`s : Π i, Submonoid f i`, `pi I s` is the submonoid of dependent functions `f : Π i, f i` such that
`f i` belongs to `Pi I s` whenever `i ∈ I`. -/
@[to_additive "A version of `Set.pi` for `AddSubmonoid`s. Given an index set `I` and a family
  of submodules `s : Π i, AddSubmonoid f i`, `pi I s` is the `AddSubmonoid` of dependent functions
  `f : Π i, f i` such that `f i` belongs to `pi I s` whenever `i ∈ I`."]
def _root_.Submonoid.pi [∀ i, MulOneClass (f i)] (I : Set η) (s : ∀ i, Submonoid (f i)) :
    Submonoid (∀ i, f i) where
  carrier := I.pi fun i => (s i).carrier
  one_mem' i _ := (s i).one_mem
  mul_mem' hp hq i hI := (s i).mul_mem (hp i hI) (hq i hI)


/-- A version of `Set.pi` for subgroups. Given an index set `I` and a family of submodules
`s : Π i, Subgroup f i`, `pi I s` is the subgroup of dependent functions `f : Π i, f i` such that
`f i` belongs to `pi I s` whenever `i ∈ I`. -/
@[to_additive
      "A version of `Set.pi` for `AddSubgroup`s. Given an index set `I` and a family
      of submodules `s : Π i, AddSubgroup f i`, `pi I s` is the `AddSubgroup` of dependent functions
      `f : Π i, f i` such that `f i` belongs to `pi I s` whenever `i ∈ I`."]
def pi (I : Set η) (H : ∀ i, Subgroup (f i)) : Subgroup (∀ i, f i) :=
  { Submonoid.pi I fun i => (H i).toSubmonoid with
    inv_mem' := fun hp i hI => (H i).inv_mem (hp i hI) }


@[to_additive]
theorem coe_pi (I : Set η) (H : ∀ i, Subgroup (f i)) :
    (pi I H : Set (∀ i, f i)) = Set.pi I fun i => (H i : Set (f i)) :=
  rfl


@[to_additive]
theorem mem_pi (I : Set η) {H : ∀ i, Subgroup (f i)} {p : ∀ i, f i} :
    p ∈ pi I H ↔ ∀ i : η, i ∈ I → p i ∈ H i :=
  Iff.rfl


@[to_additive]
theorem pi_top (I : Set η) : (pi I fun i => (⊤ : Subgroup (f i))) = ⊤ :=
                  /-
                    η : Type u_7
                    f : η → Type u_8
                    inst✝ : (i : η) → Group (f i)
                    I : Set η
                    x : (i : η) → f i
                    ⊢ Iff (Membership.mem (Subgroup.pi I fun i => Top.top) x) (Membership.mem Top. …
                  -/
  ext fun x => by simp [mem_pi]
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem pi_empty (H : ∀ i, Subgroup (f i)) : pi ∅ H = ⊤ :=
                  /-
                    η : Type u_7
                    f : η → Type u_8
                    inst✝ : (i : η) → Group (f i)
                    H : (i : η) → Subgroup (f i)
                    x : (i : η) → f i
                    ⊢ Iff (Membership.mem (Subgroup.pi EmptyCollection.emptyCollection H) x) (Memb …
                  -/
  ext fun x => by simp [mem_pi]
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem pi_bot : (pi Set.univ fun i => (⊥ : Subgroup (f i))) = ⊥ :=
  (eq_bot_iff_forall _).mpr fun p hp => by
    /-
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      p : (i : η) → f i
      hp : Membership.mem (Subgroup.pi Set.univ fun i => Bot.bot) p
      ⊢ Eq p 1
    -/
    simp only [mem_pi, mem_bot] at *
    /-
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      p : (i : η) → f i
      hp : ∀ (i : η), Membership.mem Set.univ i → Eq (p i) 1
      ⊢ Eq p 1
    -/
    ext j
    /-
      case h
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      p : (i : η) → f i
      hp : ∀ (i : η), Membership.mem Set.univ i → Eq (p i) 1
      j : η
      ⊢ Eq (p j) (1 j)
    -/
    exact hp j trivial
    /-
      🎉 no goals
    -/


@[to_additive]
theorem le_pi_iff {I : Set η} {H : ∀ i, Subgroup (f i)} {J : Subgroup (∀ i, f i)} :
    J ≤ pi I H ↔ ∀ i : η, i ∈ I → map (Pi.evalMonoidHom f i) J ≤ H i := by
  /-
    η : Type u_7
    f : η → Type u_8
    inst✝ : (i : η) → Group (f i)
    I : Set η
    H : (i : η) → Subgroup (f i)
    J : Subgroup ((i : η) → f i)
    ⊢ Iff (LE.le J (Subgroup.pi I H)) (∀ (i : η), Membership.mem I i → LE.le (Subg …
  -/
  constructor
    /-
      case mp
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      I : Set η
      H : (i : η) → Subgroup (f i)
      J : Subgroup ((i : η) → f i)
      ⊢ LE.le J (Subgroup.pi I H) → ∀ (i : η), Membership.mem I i → LE.le (Subgroup. …
    -/
  · intro h i hi
    /-
      case mp
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      I : Set η
      H : (i : η) → Subgroup (f i)
      J : Subgroup ((i : η) → f i)
      h : LE.le J (Subgroup.pi I H)
      i : η
      hi : Membership.mem I i
      ⊢ LE.le (Subgroup.map (Pi.evalMonoidHom f i) J) (H i)
    -/
    rintro _ ⟨x, hx, rfl⟩
    /-
      case mp.intro.intro
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      I : Set η
      H : (i : η) → Subgroup (f i)
      J : Subgroup ((i : η) → f i)
      h : LE.le J (Subgroup.pi I H)
      i : η
      hi : Membership.mem I i
      x : (i : η) → f i
      hx : Membership.mem (↑J) x
      ⊢ Membership.mem (H i) ((Pi.evalMonoidHom f i) x)
    -/
    exact (h hx) _ hi
    /-
      🎉 no goals
    -/
    /-
      case mpr
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      I : Set η
      H : (i : η) → Subgroup (f i)
      J : Subgroup ((i : η) → f i)
      ⊢ (∀ (i : η), Membership.mem I i → LE.le (Subgroup.map (Pi.evalMonoidHom f i)  …
    -/
  · intro h x hx i hi
    /-
      case mpr
      η : Type u_7
      f : η → Type u_8
      inst✝ : (i : η) → Group (f i)
      I : Set η
      H : (i : η) → Subgroup (f i)
      J : Subgroup ((i : η) → f i)
      h : ∀ (i : η), Membership.mem I i → LE.le (Subgroup.map (Pi.evalMonoidHom f i) …
      x : (i : η) → f i
      hx : Membership.mem J x
      i : η
      hi : Membership.mem I i
      ⊢ Membership.mem ((fun i => ((fun i => (H i).toSubmonoid) i).carrier) i) (x i)
    -/
    exact h i hi ⟨_, hx, rfl⟩
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem mulSingle_mem_pi [DecidableEq η] {I : Set η} {H : ∀ i, Subgroup (f i)} (i : η) (x : f i) :
    Pi.mulSingle i x ∈ pi I H ↔ i ∈ I → x ∈ H i := by
  /-
    η : Type u_7
    f : η → Type u_8
    inst✝¹ : (i : η) → Group (f i)
    inst✝ : DecidableEq η
    I : Set η
    H : (i : η) → Subgroup (f i)
    i : η
    x : f i
    ⊢ Iff (Membership.mem (Subgroup.pi I H) (Pi.mulSingle i x)) (Membership.mem I  …
  -/
  constructor
    /-
      case mp
      η : Type u_7
      f : η → Type u_8
      inst✝¹ : (i : η) → Group (f i)
      inst✝ : DecidableEq η
      I : Set η
      H : (i : η) → Subgroup (f i)
      i : η
      x : f i
      ⊢ Membership.mem (Subgroup.pi I H) (Pi.mulSingle i x) → Membership.mem I i → M …
    -/
  · intro h hi
    /-
      case mp
      η : Type u_7
      f : η → Type u_8
      inst✝¹ : (i : η) → Group (f i)
      inst✝ : DecidableEq η
      I : Set η
      H : (i : η) → Subgroup (f i)
      i : η
      x : f i
      h : Membership.mem (Subgroup.pi I H) (Pi.mulSingle i x)
      hi : Membership.mem I i
      ⊢ Membership.mem (H i) x
    -/
    simpa using h i hi
    /-
      🎉 no goals
    -/
    /-
      case mpr
      η : Type u_7
      f : η → Type u_8
      inst✝¹ : (i : η) → Group (f i)
      inst✝ : DecidableEq η
      I : Set η
      H : (i : η) → Subgroup (f i)
      i : η
      x : f i
      ⊢ (Membership.mem I i → Membership.mem (H i) x) → Membership.mem (Subgroup.pi  …
    -/
  · intro h j hj
    /-
      case mpr
      η : Type u_7
      f : η → Type u_8
      inst✝¹ : (i : η) → Group (f i)
      inst✝ : DecidableEq η
      I : Set η
      H : (i : η) → Subgroup (f i)
      i : η
      x : f i
      h : Membership.mem I i → Membership.mem (H i) x
      j : η
      hj : Membership.mem I j
      ⊢ Membership.mem ((fun i => ((fun i => (H i).toSubmonoid) i).carrier) j) (Pi.m …
    -/
    by_cases heq : j = i
      /-
        case pos
        η : Type u_7
        f : η → Type u_8
        inst✝¹ : (i : η) → Group (f i)
        inst✝ : DecidableEq η
        I : Set η
        H : (i : η) → Subgroup (f i)
        i : η
        x : f i
        h : Membership.mem I i → Membership.mem (H i) x
        j : η
        hj : Membership.mem I j
        heq : Eq j i
        ⊢ Membership.mem ((fun i => ((fun i => (H i).toSubmonoid) i).carrier) j) (Pi.m …
      -/
    · subst heq
      /-
        case pos
        η : Type u_7
        f : η → Type u_8
        inst✝¹ : (i : η) → Group (f i)
        inst✝ : DecidableEq η
        I : Set η
        H : (i : η) → Subgroup (f i)
        j : η
        hj : Membership.mem I j
        x : f j
        h : Membership.mem I j → Membership.mem (H j) x
        ⊢ Membership.mem ((fun i => ((fun i => (H i).toSubmonoid) i).carrier) j) (Pi.m …
      -/
      simpa using h hj
      /-
        🎉 no goals
      -/
      /-
        case neg
        η : Type u_7
        f : η → Type u_8
        inst✝¹ : (i : η) → Group (f i)
        inst✝ : DecidableEq η
        I : Set η
        H : (i : η) → Subgroup (f i)
        i : η
        x : f i
        h : Membership.mem I i → Membership.mem (H i) x
        j : η
        hj : Membership.mem I j
        heq : Not (Eq j i)
        ⊢ Membership.mem ((fun i => ((fun i => (H i).toSubmonoid) i).carrier) j) (Pi.m …
      -/
    · simp [heq, one_mem]
      /-
        🎉 no goals
      -/


@[to_additive]
theorem pi_eq_bot_iff (H : ∀ i, Subgroup (f i)) : pi Set.univ H = ⊥ ↔ ∀ i, H i = ⊥ := by
  classical
    simp only [eq_bot_iff_forall]
    constructor
    · intro h i x hx
      have : MonoidHom.mulSingle f i x = 1 :=
        h (MonoidHom.mulSingle f i x) ((mulSingle_mem_pi i x).mpr fun _ => hx)
      simpa using congr_fun this i
    · exact fun h x hx => funext fun i => h _ _ (hx i trivial)


/-- A subgroup is characteristic if it is fixed by all automorphisms.
  Several equivalent conditions are provided by lemmas of the form `Characteristic.iff...` -/
structure Characteristic : Prop where
  /-- `H` is fixed by all automorphisms -/
  fixed : ∀ ϕ : G ≃* G, H.comap ϕ.toMonoidHom = H


instance (priority := 100) normal_of_characteristic [h : H.Characteristic] : H.Normal :=
  ⟨fun a ha b => (SetLike.ext_iff.mp (h.fixed (MulAut.conj b)) a).mpr ha⟩


/-- An `AddSubgroup` is characteristic if it is fixed by all automorphisms.
  Several equivalent conditions are provided by lemmas of the form `Characteristic.iff...` -/
structure Characteristic : Prop where
  /-- `H` is fixed by all automorphisms -/
  fixed : ∀ ϕ : A ≃+ A, H.comap ϕ.toAddMonoidHom = H


instance (priority := 100) normal_of_characteristic [h : H.Characteristic] : H.Normal :=
  ⟨fun a ha b => (SetLike.ext_iff.mp (h.fixed (AddAut.conj b)) a).mpr ha⟩


@[to_additive]
theorem characteristic_iff_comap_eq : H.Characteristic ↔ ∀ ϕ : G ≃* G, H.comap ϕ.toMonoidHom = H :=
  ⟨Characteristic.fixed, Characteristic.mk⟩


@[to_additive]
theorem characteristic_iff_comap_le : H.Characteristic ↔ ∀ ϕ : G ≃* G, H.comap ϕ.toMonoidHom ≤ H :=
  characteristic_iff_comap_eq.trans
    ⟨fun h ϕ => le_of_eq (h ϕ), fun h ϕ =>
      le_antisymm (h ϕ) fun g hg => h ϕ.symm ((congr_arg (· ∈ H) (ϕ.symm_apply_apply g)).mpr hg)⟩


@[to_additive]
theorem characteristic_iff_le_comap : H.Characteristic ↔ ∀ ϕ : G ≃* G, H ≤ H.comap ϕ.toMonoidHom :=
  characteristic_iff_comap_eq.trans
    ⟨fun h ϕ => ge_of_eq (h ϕ), fun h ϕ =>
      le_antisymm (fun g hg => (congr_arg (· ∈ H) (ϕ.symm_apply_apply g)).mp (h ϕ.symm hg)) (h ϕ)⟩


@[to_additive]
theorem characteristic_iff_map_eq : H.Characteristic ↔ ∀ ϕ : G ≃* G, H.map ϕ.toMonoidHom = H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), Eq (Subgroup.map ϕ.toMonoidHom H …
  -/
  simp_rw [map_equiv_eq_comap_symm']
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), Eq (Subgroup.comap ϕ.symm.toMono …
  -/
  exact characteristic_iff_comap_eq.trans ⟨fun h ϕ => h ϕ.symm, fun h ϕ => h ϕ.symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem characteristic_iff_map_le : H.Characteristic ↔ ∀ ϕ : G ≃* G, H.map ϕ.toMonoidHom ≤ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), LE.le (Subgroup.map ϕ.toMonoidHo …
  -/
  simp_rw [map_equiv_eq_comap_symm']
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), LE.le (Subgroup.comap ϕ.symm.toM …
  -/
  exact characteristic_iff_comap_le.trans ⟨fun h ϕ => h ϕ.symm, fun h ϕ => h ϕ.symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem characteristic_iff_le_map : H.Characteristic ↔ ∀ ϕ : G ≃* G, H ≤ H.map ϕ.toMonoidHom := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), LE.le H (Subgroup.map ϕ.toMonoid …
  -/
  simp_rw [map_equiv_eq_comap_symm']
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff H.Characteristic (∀ (ϕ : MulEquiv G G), LE.le H (Subgroup.comap ϕ.symm.t …
  -/
  exact characteristic_iff_le_comap.trans ⟨fun h ϕ => h ϕ.symm, fun h ϕ => h ϕ.symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
instance botCharacteristic : Characteristic (⊥ : Subgroup G) :=
  characteristic_iff_le_map.mpr fun _ϕ => bot_le


@[to_additive]
instance topCharacteristic : Characteristic (⊤ : Subgroup G) :=
  characteristic_iff_map_le.mpr fun _ϕ => le_top



@[to_additive]
instance (priority := 100) normal_in_normalizer : (H.subgroupOf H.normalizer).Normal :=
                    /-
                      G : Type u_1
                      G' : Type u_2
                      G'' : Type u_3
                      inst✝³ : Group G
                      inst✝² : Group G'
                      inst✝¹ : Group G''
                      A : Type u_4
                      inst✝ : AddGroup A
                      H K : Subgroup G
                      x : Subtype fun x => Membership.mem H.normalizer x
                      xH : Membership.mem (H.subgroupOf H.normalizer) x
                      g : Subtype fun x => Membership.mem H.normalizer x
                      ⊢ Membership.mem (H.subgroupOf H.normalizer) (HMul.hMul (HMul.hMul g x) (Inv.i …
                    -/
  ⟨fun x xH g => by simpa only [mem_subgroupOf] using (g.2 x.1).1 xH⟩
                    /-
                      🎉 no goals
                    -/


@[to_additive]
theorem normalizer_eq_top_iff : H.normalizer = ⊤ ↔ H.Normal :=
  eq_top_iff.trans
    ⟨fun h => ⟨fun a ha b => (h (mem_top b) a).mp ha⟩, fun h a _ha b =>
                                                 /-
                                                   G : Type u_1
                                                   inst✝ : Group G
                                                   H : Subgroup G
                                                   h : H.Normal
                                                   a : G
                                                   _ha : Membership.mem Top.top a
                                                   b : G
                                                   hb : Membership.mem H (HMul.hMul (HMul.hMul a b) (Inv.inv a))
                                                   ⊢ Membership.mem H b
                                                 -/
      ⟨fun hb => h.conj_mem b hb a, fun hb => by rwa [h.mem_comm_iff, inv_mul_cancel_left] at hb⟩⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


variable (H) in
@[to_additive]
theorem normalizer_eq_top [h : H.Normal] : H.normalizer = ⊤ :=
  normalizer_eq_top_iff.mpr h


@[to_additive]
theorem le_normalizer_of_normal [hK : (H.subgroupOf K).Normal] (HK : H ≤ K) : K ≤ H.normalizer :=
  fun x hx y =>
  ⟨fun yH => hK.conj_mem ⟨y, HK yH⟩ yH ⟨x, hx⟩, fun yH => by
    simpa [mem_subgroupOf, mul_assoc] using
      hK.conj_mem ⟨x * y * x⁻¹, HK yH⟩ yH ⟨x⁻¹, K.inv_mem hx⟩⟩


/-- The preimage of the normalizer is contained in the normalizer of the preimage. -/
@[to_additive "The preimage of the normalizer is contained in the normalizer of the preimage."]
theorem le_normalizer_comap (f : N →* G) :
    H.normalizer.comap f ≤ (H.comap f).normalizer := fun x => by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom N G
    x : N
    ⊢ Membership.mem (Subgroup.comap f H.normalizer) x → Membership.mem (Subgroup. …
  -/
  simp only [mem_normalizer_iff, mem_comap]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom N G
    x : N
    ⊢ (∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hMul …
  -/
  intro h n
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom N G
    x : N
    h : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hMu …
    n : N
    ⊢ Iff (Membership.mem H (f n)) (Membership.mem H (f (HMul.hMul (HMul.hMul x n) …
  -/
  simp [h (f n)]
  /-
    🎉 no goals
  -/


/-- The image of the normalizer is contained in the normalizer of the image. -/
@[to_additive "The image of the normalizer is contained in the normalizer of the image."]
theorem le_normalizer_map (f : G →* N) : H.normalizer.map f ≤ (H.map f).normalizer := fun _ => by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    x✝ : N
    ⊢ Membership.mem (Subgroup.map f H.normalizer) x✝ → Membership.mem (Subgroup.m …
  -/
  simp only [and_imp, exists_prop, mem_map, exists_imp, mem_normalizer_iff]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    x✝ : N
    ⊢ ∀ (x : G), (∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul …
  -/
  rintro x hx rfl n
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    x : G
    hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
    n : N
    ⊢ Iff (Exists fun x => And (Membership.mem H x) (Eq (f x) n)) (Exists fun x_1  …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      n : N
      ⊢ (Exists fun x => And (Membership.mem H x) (Eq (f x) n)) → Exists fun x_1 =>  …
    -/
  · rintro ⟨y, hy, rfl⟩
    /-
      case mp.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      y : G
      hy : Membership.mem H y
      ⊢ Exists fun x_1 => And (Membership.mem H x_1) (Eq (f x_1) (HMul.hMul (HMul.hM …
    -/
    use x * y * x⁻¹, (hx y).1 hy
    /-
      case right
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      y : G
      hy : Membership.mem H y
      ⊢ Eq (f (HMul.hMul (HMul.hMul x y) (Inv.inv x))) (HMul.hMul (HMul.hMul (f x) ( …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      n : N
      ⊢ (Exists fun x_1 => And (Membership.mem H x_1) (Eq (f x_1) (HMul.hMul (HMul.h …
    -/
  · rintro ⟨y, hyH, hy⟩
    /-
      case mpr.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      n : N
      y : G
      hyH : Membership.mem H y
      hy : Eq (f y) (HMul.hMul (HMul.hMul (f x) n) (Inv.inv (f x)))
      ⊢ Exists fun x => And (Membership.mem H x) (Eq (f x) n)
    -/
    use x⁻¹ * y * x
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      n : N
      y : G
      hyH : Membership.mem H y
      hy : Eq (f y) (HMul.hMul (HMul.hMul (f x) n) (Inv.inv (f x)))
      ⊢ And (Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) y) x)) (Eq (f (HMul. …
    -/
    rw [hx]
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      H : Subgroup G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      x : G
      hx : ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hM …
      n : N
      y : G
      hyH : Membership.mem H y
      hy : Eq (f y) (HMul.hMul (HMul.hMul (f x) n) (Inv.inv (f x)))
      ⊢ And (Membership.mem H (HMul.hMul (HMul.hMul x (HMul.hMul (HMul.hMul (Inv.inv …
    -/
    simp [hy, hyH, mul_assoc]
    /-
      🎉 no goals
    -/


/-- Every proper subgroup `H` of `G` is a proper normal subgroup of the normalizer of `H` in `G`. -/
def _root_.NormalizerCondition :=
  ∀ H : Subgroup G, H < ⊤ → H < normalizer H


/-- Alternative phrasing of the normalizer condition: Only the full group is self-normalizing.
This may be easier to work with, as it avoids inequalities and negations. -/
theorem _root_.normalizerCondition_iff_only_full_group_self_normalizing :
    NormalizerCondition G ↔ ∀ H : Subgroup G, H.normalizer = H → H = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Iff (NormalizerCondition G) (∀ (H : Subgroup G), Eq H.normalizer H → Eq H To …
  -/
  apply forall_congr'; intro H
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (LT.lt H Top.top → LT.lt H H.normalizer) (Eq H.normalizer H → Eq H Top.t …
  -/
  simp only [lt_iff_le_and_ne, le_normalizer, le_top, Ne]
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (And True (Not (Eq H Top.top)) → And True (Not (Eq H H.normalizer))) (Eq …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- Given a set `s`, `conjugatesOfSet s` is the set of all conjugates of
the elements of `s`. -/
def conjugatesOfSet (s : Set G) : Set G :=
  ⋃ a ∈ s, conjugatesOf a


theorem mem_conjugatesOfSet_iff {x : G} : x ∈ conjugatesOfSet s ↔ ∃ a ∈ s, IsConj a x := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    x : G
    ⊢ Iff (Membership.mem (Group.conjugatesOfSet s) x) (Exists fun a => And (Membe …
  -/
  erw [Set.mem_iUnion₂]; simp only [conjugatesOf, isConj_iff, Set.mem_setOf_eq, exists_prop]
                         /-
                           🎉 no goals
                         -/


theorem subset_conjugatesOfSet : s ⊆ conjugatesOfSet s := fun (x : G) (h : x ∈ s) =>
  mem_conjugatesOfSet_iff.2 ⟨x, h, IsConj.refl _⟩


theorem conjugatesOfSet_mono {s t : Set G} (h : s ⊆ t) : conjugatesOfSet s ⊆ conjugatesOfSet t :=
  Set.biUnion_subset_biUnion_left h


theorem conjugates_subset_normal {N : Subgroup G} [tn : N.Normal] {a : G} (h : a ∈ N) :
    conjugatesOf a ⊆ N := by
  /-
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    tn : N.Normal
    a : G
    h : Membership.mem N a
    ⊢ HasSubset.Subset (conjugatesOf a) ↑N
  -/
  rintro a hc
  /-
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    tn : N.Normal
    a✝ : G
    h : Membership.mem N a✝
    a : G
    hc : Membership.mem (conjugatesOf a✝) a
    ⊢ Membership.mem (↑N) a
  -/
  obtain ⟨c, rfl⟩ := isConj_iff.1 hc
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    tn : N.Normal
    a : G
    h : Membership.mem N a
    c : G
    hc : Membership.mem (conjugatesOf a) (HMul.hMul (HMul.hMul c a) (Inv.inv c))
    ⊢ Membership.mem (↑N) (HMul.hMul (HMul.hMul c a) (Inv.inv c))
  -/
  exact tn.conj_mem a h c
  /-
    🎉 no goals
  -/


theorem conjugatesOfSet_subset {s : Set G} {N : Subgroup G} [N.Normal] (h : s ⊆ N) :
    conjugatesOfSet s ⊆ N :=
  Set.iUnion₂_subset fun _x H => conjugates_subset_normal (h H)


/-- The set of conjugates of `s` is closed under conjugation. -/
theorem conj_mem_conjugatesOfSet {x c : G} :
    x ∈ conjugatesOfSet s → c * x * c⁻¹ ∈ conjugatesOfSet s := fun H => by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    x c : G
    H : Membership.mem (Group.conjugatesOfSet s) x
    ⊢ Membership.mem (Group.conjugatesOfSet s) (HMul.hMul (HMul.hMul c x) (Inv.inv …
  -/
  rcases mem_conjugatesOfSet_iff.1 H with ⟨a, h₁, h₂⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    s : Set G
    x c : G
    H : Membership.mem (Group.conjugatesOfSet s) x
    a : G
    h₁ : Membership.mem s a
    h₂ : IsConj a x
    ⊢ Membership.mem (Group.conjugatesOfSet s) (HMul.hMul (HMul.hMul c x) (Inv.inv …
  -/
  exact mem_conjugatesOfSet_iff.2 ⟨a, h₁, h₂.trans (isConj_iff.2 ⟨c, rfl⟩)⟩
  /-
    🎉 no goals
  -/


/-- The normal closure of a set `s` is the subgroup closure of all the conjugates of
elements of `s`. It is the smallest normal subgroup containing `s`. -/
def normalClosure (s : Set G) : Subgroup G :=
  closure (conjugatesOfSet s)


theorem conjugatesOfSet_subset_normalClosure : conjugatesOfSet s ⊆ normalClosure s :=
  subset_closure


theorem subset_normalClosure : s ⊆ normalClosure s :=
  Set.Subset.trans subset_conjugatesOfSet conjugatesOfSet_subset_normalClosure


theorem le_normalClosure {H : Subgroup G} : H ≤ normalClosure ↑H := fun _ h =>
  subset_normalClosure h


/-- The normal closure of `s` is a normal subgroup. -/
instance normalClosure_normal : (normalClosure s).Normal :=
  ⟨fun n h g => by
    refine Subgroup.closure_induction (fun x hx => ?_) ?_ (fun x y _ _ ihx ihy => ?_)
      (fun x _ ihx => ?_) h
      /-
        case refine_1
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g x : G
        hx : Membership.mem (Group.conjugatesOfSet s) x
        ⊢ Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g x) (Inv.in …
      -/
    · exact conjugatesOfSet_subset_normalClosure (conj_mem_conjugatesOfSet hx)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g : G
        ⊢ Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g 1) (Inv.in …
      -/
    · simpa using (normalClosure s).one_mem
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g x y : G
        x✝¹ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
        x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) y
        ihx : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g x) (In …
        ihy : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g y) (In …
        ⊢ Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g (HMul.hMul …
      -/
    · rw [← conj_mul]
      /-
        case refine_3
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g x y : G
        x✝¹ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
        x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) y
        ihx : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g x) (In …
        ihy : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g y) (In …
        ⊢ Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul (HMul.hMul g …
      -/
      exact mul_mem ihx ihy
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g x : G
        x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
        ihx : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g x) (In …
        ⊢ Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g (Inv.inv x …
      -/
    · rw [← conj_inv]
      /-
        case refine_4
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝³ : Group G
        inst✝² : Group G'
        inst✝¹ : Group G''
        A : Type u_4
        inst✝ : AddGroup A
        s : Set G
        n : G
        h : Membership.mem (Subgroup.normalClosure s) n
        g x : G
        x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
        ihx : Membership.mem (Subgroup.normalClosure s) (HMul.hMul (HMul.hMul g x) (In …
        ⊢ Membership.mem (Subgroup.normalClosure s) (Inv.inv (HMul.hMul (HMul.hMul g x …
      -/
      exact inv_mem ihx⟩
      /-
        🎉 no goals
      -/


/-- The normal closure of `s` is the smallest normal subgroup containing `s`. -/
theorem normalClosure_le_normal {N : Subgroup G} [N.Normal] (h : s ⊆ N) : normalClosure s ≤ N := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    s : Set G
    N : Subgroup G
    inst✝ : N.Normal
    h : HasSubset.Subset s ↑N
    ⊢ LE.le (Subgroup.normalClosure s) N
  -/
  intro a w
  /-
    G : Type u_1
    inst✝¹ : Group G
    s : Set G
    N : Subgroup G
    inst✝ : N.Normal
    h : HasSubset.Subset s ↑N
    a : G
    w : Membership.mem (Subgroup.normalClosure s) a
    ⊢ Membership.mem N a
  -/
  refine closure_induction (fun x hx => ?_) ?_ (fun x y _ _ ihx ihy => ?_) (fun x _ ihx => ?_) w
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Group G
      s : Set G
      N : Subgroup G
      inst✝ : N.Normal
      h : HasSubset.Subset s ↑N
      a : G
      w : Membership.mem (Subgroup.normalClosure s) a
      x : G
      hx : Membership.mem (Group.conjugatesOfSet s) x
      ⊢ Membership.mem N x
    -/
  · exact conjugatesOfSet_subset h hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      s : Set G
      N : Subgroup G
      inst✝ : N.Normal
      h : HasSubset.Subset s ↑N
      a : G
      w : Membership.mem (Subgroup.normalClosure s) a
      ⊢ Membership.mem N 1
    -/
  · exact one_mem _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝¹ : Group G
      s : Set G
      N : Subgroup G
      inst✝ : N.Normal
      h : HasSubset.Subset s ↑N
      a : G
      w : Membership.mem (Subgroup.normalClosure s) a
      x y : G
      x✝¹ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
      x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) y
      ihx : Membership.mem N x
      ihy : Membership.mem N y
      ⊢ Membership.mem N (HMul.hMul x y)
    -/
  · exact mul_mem ihx ihy
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      G : Type u_1
      inst✝¹ : Group G
      s : Set G
      N : Subgroup G
      inst✝ : N.Normal
      h : HasSubset.Subset s ↑N
      a : G
      w : Membership.mem (Subgroup.normalClosure s) a
      x : G
      x✝ : Membership.mem (Subgroup.closure (Group.conjugatesOfSet s)) x
      ihx : Membership.mem N x
      ⊢ Membership.mem N (Inv.inv x)
    -/
  · exact inv_mem ihx
    /-
      🎉 no goals
    -/


theorem normalClosure_subset_iff {N : Subgroup G} [N.Normal] : s ⊆ N ↔ normalClosure s ≤ N :=
  ⟨normalClosure_le_normal, Set.Subset.trans subset_normalClosure⟩


@[gcongr]
theorem normalClosure_mono {s t : Set G} (h : s ⊆ t) : normalClosure s ≤ normalClosure t :=
  normalClosure_le_normal (Set.Subset.trans h subset_normalClosure)


theorem normalClosure_eq_iInf :
    normalClosure s = ⨅ (N : Subgroup G) (_ : Normal N) (_ : s ⊆ N), N :=
  le_antisymm (le_iInf fun _ => le_iInf fun _ => le_iInf normalClosure_le_normal)
    (iInf_le_of_le (normalClosure s)
                         /-
                           G : Type u_1
                           inst✝ : Group G
                           s : Set G
                           ⊢ (Subgroup.normalClosure s).Normal
                         -/
      (iInf_le_of_le (by infer_instance) (iInf_le_of_le subset_normalClosure le_rfl)))
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem normalClosure_eq_self (H : Subgroup G) [H.Normal] : normalClosure ↑H = H :=
  le_antisymm (normalClosure_le_normal rfl.subset) le_normalClosure


theorem normalClosure_idempotent : normalClosure ↑(normalClosure s) = normalClosure s :=
  normalClosure_eq_self _


theorem closure_le_normalClosure {s : Set G} : closure s ≤ normalClosure s := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    ⊢ LE.le (Subgroup.closure s) (Subgroup.normalClosure s)
  -/
  simp only [subset_normalClosure, closure_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem normalClosure_closure_eq_normalClosure {s : Set G} :
    normalClosure ↑(closure s) = normalClosure s :=
  le_antisymm (normalClosure_le_normal closure_le_normalClosure) (normalClosure_mono subset_closure)


/-- The normal core of a subgroup `H` is the largest normal subgroup of `G` contained in `H`,
as shown by `Subgroup.normalCore_eq_iSup`. -/
def normalCore (H : Subgroup G) : Subgroup G where
  carrier := { a : G | ∀ b : G, b * a * b⁻¹ ∈ H }
                   /-
                     G : Type u_1
                     G' : Type u_2
                     G'' : Type u_3
                     inst✝³ : Group G
                     inst✝² : Group G'
                     inst✝¹ : Group G''
                     A : Type u_4
                     inst✝ : AddGroup A
                     s : Set G
                     H : Subgroup G
                     a : G
                     ⊢ Membership.mem H (HMul.hMul (HMul.hMul a 1) (Inv.inv a))
                   -/
  one_mem' a := by rw [mul_one, mul_inv_cancel]; exact H.one_mem
                                                 /-
                                                   🎉 no goals
                                                 -/
  inv_mem' {_} h b := (congr_arg (· ∈ H) conj_inv).mp (H.inv_mem (h b))
  mul_mem' {_ _} ha hb c := (congr_arg (· ∈ H) conj_mul).mp (H.mul_mem (ha c) (hb c))


theorem normalCore_le (H : Subgroup G) : H.normalCore ≤ H := fun a h => by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a : G
    h : Membership.mem H.normalCore a
    ⊢ Membership.mem H a
  -/
  rw [← mul_one a, ← inv_one, ← one_mul a]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    a : G
    h : Membership.mem H.normalCore a
    ⊢ Membership.mem H (HMul.hMul (HMul.hMul 1 a) (Inv.inv 1))
  -/
  exact h 1
  /-
    🎉 no goals
  -/


instance normalCore_normal (H : Subgroup G) : H.normalCore.Normal :=
  ⟨fun a h b c => by
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝³ : Group G
      inst✝² : Group G'
      inst✝¹ : Group G''
      A : Type u_4
      inst✝ : AddGroup A
      s : Set G
      H : Subgroup G
      a : G
      h : Membership.mem H.normalCore a
      b c : G
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul c (HMul.hMul (HMul.hMul b a) (Inv.inv …
    -/
    rw [mul_assoc, mul_assoc, ← mul_inv_rev, ← mul_assoc, ← mul_assoc]; exact h (c * b)⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem normal_le_normalCore {H : Subgroup G} {N : Subgroup G} [hN : N.Normal] :
    N ≤ H.normalCore ↔ N ≤ H :=
  ⟨ge_trans H.normalCore_le, fun h_le n hn g => h_le (hN.conj_mem n hn g)⟩


theorem normalCore_mono {H K : Subgroup G} (h : H ≤ K) : H.normalCore ≤ K.normalCore :=
  normal_le_normalCore.mpr (H.normalCore_le.trans h)


theorem normalCore_eq_iSup (H : Subgroup G) :
    H.normalCore = ⨆ (N : Subgroup G) (_ : Normal N) (_ : N ≤ H), N :=
  le_antisymm
    (le_iSup_of_le H.normalCore
      (le_iSup_of_le H.normalCore_normal (le_iSup_of_le H.normalCore_le le_rfl)))
    (iSup_le fun _ => iSup_le fun _ => iSup_le normal_le_normalCore.mpr)


@[simp]
theorem normalCore_eq_self (H : Subgroup G) [H.Normal] : H.normalCore = H :=
  le_antisymm H.normalCore_le (normal_le_normalCore.mpr le_rfl)


theorem normalCore_idempotent (H : Subgroup G) : H.normalCore.normalCore = H.normalCore :=
  H.normalCore.normalCore_eq_self


@[to_additive]
theorem prodMap_comap_prod {G' : Type*} {N' : Type*} [Group G'] [Group N'] (f : G →* N)
    (g : G' →* N') (S : Subgroup N) (S' : Subgroup N') :
    (S.prod S').comap (prodMap f g) = (S.comap f).prod (S'.comap g) :=
  SetLike.coe_injective <| Set.preimage_prod_map_prod f g _ _


@[to_additive]
theorem ker_prodMap {G' : Type*} {N' : Type*} [Group G'] [Group N'] (f : G →* N) (g : G' →* N') :
    (prodMap f g).ker = f.ker.prod g.ker := by
  /-
    G : Type u_1
    inst✝³ : Group G
    N : Type u_5
    inst✝² : Group N
    G' : Type u_8
    N' : Type u_9
    inst✝¹ : Group G'
    inst✝ : Group N'
    f : MonoidHom G N
    g : MonoidHom G' N'
    ⊢ Eq (f.prodMap g).ker (f.ker.prod g.ker)
  -/
  rw [← comap_bot, ← comap_bot, ← comap_bot, ← prodMap_comap_prod, bot_prod_bot]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma ker_fst : ker (fst G G') = .prod ⊥ ⊤ := SetLike.ext fun _ => (iff_of_eq (and_true _)).symm


@[to_additive (attr := simp)]
lemma ker_snd : ker (snd G G') = .prod ⊤ ⊥ := SetLike.ext fun _ => (iff_of_eq (true_and _)).symm


@[to_additive]
theorem Normal.map {H : Subgroup G} (h : H.Normal) (f : G →* N) (hf : Function.Surjective f) :
    (H.map f).Normal := by
  rw [← normalizer_eq_top_iff, ← top_le_iff, ← f.range_eq_top_of_surjective hf, f.range_eq_map,
    ← H.normalizer_eq_top]
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    h : H.Normal
    f : MonoidHom G N
    hf : Function.Surjective ⇑f
    ⊢ LE.le (Subgroup.map f H.normalizer) (Subgroup.map f H).normalizer
  -/
  exact le_normalizer_map _
  /-
    🎉 no goals
  -/


/-- The preimage of the normalizer is equal to the normalizer of the preimage of a surjective
  function. -/
@[to_additive
      "The preimage of the normalizer is equal to the normalizer of the preimage of
      a surjective function."]
theorem comap_normalizer_eq_of_surjective (H : Subgroup G) {f : N →* G}
    (hf : Function.Surjective f) : H.normalizer.comap f = (H.comap f).normalizer :=
  le_antisymm (le_normalizer_comap f)
    (by
      /-
        G : Type u_1
        inst✝¹ : Group G
        N : Type u_5
        inst✝ : Group N
        H : Subgroup G
        f : MonoidHom N G
        hf : Function.Surjective ⇑f
        ⊢ LE.le (Subgroup.comap f H).normalizer (Subgroup.comap f H.normalizer)
      -/
      intro x hx
      /-
        G : Type u_1
        inst✝¹ : Group G
        N : Type u_5
        inst✝ : Group N
        H : Subgroup G
        f : MonoidHom N G
        hf : Function.Surjective ⇑f
        x : N
        hx : Membership.mem (Subgroup.comap f H).normalizer x
        ⊢ Membership.mem (Subgroup.comap f H.normalizer) x
      -/
      simp only [mem_comap, mem_normalizer_iff] at *
      /-
        G : Type u_1
        inst✝¹ : Group G
        N : Type u_5
        inst✝ : Group N
        H : Subgroup G
        f : MonoidHom N G
        hf : Function.Surjective ⇑f
        x : N
        hx : ∀ (h : N), Iff (Membership.mem H (f h)) (Membership.mem H (f (HMul.hMul ( …
        ⊢ ∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul.hMul  …
      -/
      intro n
      /-
        G : Type u_1
        inst✝¹ : Group G
        N : Type u_5
        inst✝ : Group N
        H : Subgroup G
        f : MonoidHom N G
        hf : Function.Surjective ⇑f
        x : N
        hx : ∀ (h : N), Iff (Membership.mem H (f h)) (Membership.mem H (f (HMul.hMul ( …
        n : G
        ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul (f x) n) (I …
      -/
      rcases hf n with ⟨y, rfl⟩
      /-
        case intro
        G : Type u_1
        inst✝¹ : Group G
        N : Type u_5
        inst✝ : Group N
        H : Subgroup G
        f : MonoidHom N G
        hf : Function.Surjective ⇑f
        x : N
        hx : ∀ (h : N), Iff (Membership.mem H (f h)) (Membership.mem H (f (HMul.hMul ( …
        y : N
        ⊢ Iff (Membership.mem H (f y)) (Membership.mem H (HMul.hMul (HMul.hMul (f x) ( …
      -/
      simp [hx y])
      /-
        🎉 no goals
      -/


@[to_additive]
theorem comap_normalizer_eq_of_injective_of_le_range {N : Type*} [Group N] (H : Subgroup G)
    {f : N →* G} (hf : Function.Injective f) (h : H.normalizer ≤ f.range) :
    comap f H.normalizer = (comap f H).normalizer := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_6
    inst✝ : Group N
    H : Subgroup G
    f : MonoidHom N G
    hf : Function.Injective ⇑f
    h : LE.le H.normalizer f.range
    ⊢ Eq (Subgroup.comap f H.normalizer) (Subgroup.comap f H).normalizer
  -/
  apply Subgroup.map_injective hf
  /-
    case a
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_6
    inst✝ : Group N
    H : Subgroup G
    f : MonoidHom N G
    hf : Function.Injective ⇑f
    h : LE.le H.normalizer f.range
    ⊢ Eq (Subgroup.map f (Subgroup.comap f H.normalizer)) (Subgroup.map f (Subgrou …
  -/
  rw [map_comap_eq_self h]
  /-
    case a
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_6
    inst✝ : Group N
    H : Subgroup G
    f : MonoidHom N G
    hf : Function.Injective ⇑f
    h : LE.le H.normalizer f.range
    ⊢ Eq H.normalizer (Subgroup.map f (Subgroup.comap f H).normalizer)
  -/
  apply le_antisymm
    /-
      case a.a
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_6
      inst✝ : Group N
      H : Subgroup G
      f : MonoidHom N G
      hf : Function.Injective ⇑f
      h : LE.le H.normalizer f.range
      ⊢ LE.le H.normalizer (Subgroup.map f (Subgroup.comap f H).normalizer)
    -/
  · refine le_trans (le_of_eq ?_) (map_mono (le_normalizer_comap _))
    /-
      case a.a
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_6
      inst✝ : Group N
      H : Subgroup G
      f : MonoidHom N G
      hf : Function.Injective ⇑f
      h : LE.le H.normalizer f.range
      ⊢ Eq H.normalizer (Subgroup.map f (Subgroup.comap f H.normalizer))
    -/
    rw [map_comap_eq_self h]
    /-
      🎉 no goals
    -/
    /-
      case a.a
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_6
      inst✝ : Group N
      H : Subgroup G
      f : MonoidHom N G
      hf : Function.Injective ⇑f
      h : LE.le H.normalizer f.range
      ⊢ LE.le (Subgroup.map f (Subgroup.comap f H).normalizer) H.normalizer
    -/
  · refine le_trans (le_normalizer_map f) (le_of_eq ?_)
    /-
      case a.a
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_6
      inst✝ : Group N
      H : Subgroup G
      f : MonoidHom N G
      hf : Function.Injective ⇑f
      h : LE.le H.normalizer f.range
      ⊢ Eq (Subgroup.map f (Subgroup.comap f H)).normalizer H.normalizer
    -/
    rw [map_comap_eq_self (le_trans le_normalizer h)]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem subgroupOf_normalizer_eq {H N : Subgroup G} (h : H.normalizer ≤ N) :
    H.normalizer.subgroupOf N = (H.subgroupOf N).normalizer := by
  /-
    G : Type u_1
    inst✝ : Group G
    H N : Subgroup G
    h : LE.le H.normalizer N
    ⊢ Eq (H.normalizer.subgroupOf N) (H.subgroupOf N).normalizer
  -/
  apply comap_normalizer_eq_of_injective_of_le_range
    /-
      case hf
      G : Type u_1
      inst✝ : Group G
      H N : Subgroup G
      h : LE.le H.normalizer N
      ⊢ Function.Injective ⇑N.subtype
    -/
  · exact Subtype.coe_injective
    /-
      🎉 no goals
    -/
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H N : Subgroup G
    h : LE.le H.normalizer N
    ⊢ LE.le H.normalizer N.subtype.range
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- The image of the normalizer is equal to the normalizer of the image of an isomorphism. -/
@[to_additive
      "The image of the normalizer is equal to the normalizer of the image of an
      isomorphism."]
theorem map_equiv_normalizer_eq (H : Subgroup G) (f : G ≃* N) :
    H.normalizer.map f.toMonoidHom = (H.map f.toMonoidHom).normalizer := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    ⊢ Eq (Subgroup.map f.toMonoidHom H.normalizer) (Subgroup.map f.toMonoidHom H). …
  -/
  ext x
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    ⊢ Iff (Membership.mem (Subgroup.map f.toMonoidHom H.normalizer) x) (Membership …
  -/
  simp only [mem_normalizer_iff, mem_map_equiv]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    ⊢ Iff (∀ (h : G), Iff (Membership.mem H h) (Membership.mem H (HMul.hMul (HMul. …
  -/
  rw [f.toEquiv.forall_congr]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    ⊢ ∀ (a : G), Iff (Iff (Membership.mem H a) (Membership.mem H (HMul.hMul (HMul. …
  -/
  intro
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    a✝ : G
    ⊢ Iff (Iff (Membership.mem H a✝) (Membership.mem H (HMul.hMul (HMul.hMul (f.sy …
  -/
  erw [f.toEquiv.symm_apply_apply]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    a✝ : G
    ⊢ Iff (Iff (Membership.mem H a✝) (Membership.mem H (HMul.hMul (HMul.hMul (f.sy …
  -/
  simp only [map_mul, map_inv]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup G
    f : MulEquiv G N
    x : N
    a✝ : G
    ⊢ Iff (Iff (Membership.mem H a✝) (Membership.mem H (HMul.hMul (HMul.hMul (f.sy …
  -/
  erw [f.toEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- The image of the normalizer is equal to the normalizer of the image of a bijective
  function. -/
@[to_additive
      "The image of the normalizer is equal to the normalizer of the image of a bijective
        function."]
theorem map_normalizer_eq_of_bijective (H : Subgroup G) {f : G →* N} (hf : Function.Bijective f) :
    H.normalizer.map f = (H.map f).normalizer :=
  map_equiv_normalizer_eq H (MulEquiv.ofBijective f hf)


/-- Auxiliary definition used to define `liftOfRightInverse` -/
@[to_additive "Auxiliary definition used to define `liftOfRightInverse`"]
def liftOfRightInverseAux (hf : Function.RightInverse f_inv f) (g : G₁ →* G₃) (hg : f.ker ≤ g.ker) :
    G₂ →* G₃ where
  toFun b := g (f_inv b)
  map_one' := hg (hf 1)
  map_mul' := by
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : MonoidHom G₁ G₃
      hg : LE.le f.ker g.ker
      ⊢ ∀ (x y : G₂), Eq ({ toFun := fun b => g (f_inv b), map_one' := ⋯ }.toFun (HM …
    -/
    intro x y
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : MonoidHom G₁ G₃
      hg : LE.le f.ker g.ker
      x y : G₂
      ⊢ Eq ({ toFun := fun b => g (f_inv b), map_one' := ⋯ }.toFun (HMul.hMul x y))  …
    -/
    rw [← g.map_mul, ← mul_inv_eq_one, ← g.map_inv, ← g.map_mul, ← g.mem_ker]
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : MonoidHom G₁ G₃
      hg : LE.le f.ker g.ker
      x y : G₂
      ⊢ Membership.mem g.ker (HMul.hMul (f_inv (HMul.hMul x y)) (Inv.inv (HMul.hMul  …
    -/
    apply hg
    /-
      case a
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : MonoidHom G₁ G₃
      hg : LE.le f.ker g.ker
      x y : G₂
      ⊢ Membership.mem f.ker (HMul.hMul (f_inv (HMul.hMul x y)) (Inv.inv (HMul.hMul  …
    -/
    rw [f.mem_ker, f.map_mul, f.map_inv, mul_inv_eq_one, f.map_mul]
    /-
      case a
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : MonoidHom G₁ G₃
      hg : LE.le f.ker g.ker
      x y : G₂
      ⊢ Eq (f (f_inv (HMul.hMul x y))) (HMul.hMul (f (f_inv x)) (f (f_inv y)))
    -/
    simp only [hf _]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem liftOfRightInverseAux_comp_apply (hf : Function.RightInverse f_inv f) (g : G₁ →* G₃)
    (hg : f.ker ≤ g.ker) (x : G₁) : (f.liftOfRightInverseAux f_inv hf g hg) (f x) = g x := by
  /-
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    x : G₁
    ⊢ Eq ((f.liftOfRightInverseAux f_inv hf g hg) (f x)) (g x)
  -/
  dsimp [liftOfRightInverseAux]
  /-
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    x : G₁
    ⊢ Eq (g (f_inv (f x))) (g x)
  -/
  rw [← mul_inv_eq_one, ← g.map_inv, ← g.map_mul, ← g.mem_ker]
  /-
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    x : G₁
    ⊢ Membership.mem g.ker (HMul.hMul (f_inv (f x)) (Inv.inv x))
  -/
  apply hg
  /-
    case a
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    x : G₁
    ⊢ Membership.mem f.ker (HMul.hMul (f_inv (f x)) (Inv.inv x))
  -/
  rw [f.mem_ker, f.map_mul, f.map_inv, mul_inv_eq_one]
  /-
    case a
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    x : G₁
    ⊢ Eq (f (f_inv (f x))) (f x)
  -/
  simp only [hf _]
  /-
    🎉 no goals
  -/


/-- `liftOfRightInverse f hf g hg` is the unique group homomorphism `φ`

* such that `φ.comp f = g` (`MonoidHom.liftOfRightInverse_comp`),
* where `f : G₁ →+* G₂` has a RightInverse `f_inv` (`hf`),
* and `g : G₂ →+* G₃` satisfies `hg : f.ker ≤ g.ker`.

See `MonoidHom.eq_liftOfRightInverse` for the uniqueness lemma.

```
   G₁.
   |  \
 f |   \ g
   |    \
   v     \⌟
   G₂----> G₃
      ∃!φ
```
 -/
@[to_additive
      "`liftOfRightInverse f f_inv hf g hg` is the unique additive group homomorphism `φ`
      * such that `φ.comp f = g` (`AddMonoidHom.liftOfRightInverse_comp`),
      * where `f : G₁ →+ G₂` has a RightInverse `f_inv` (`hf`),
      * and `g : G₂ →+ G₃` satisfies `hg : f.ker ≤ g.ker`.
      See `AddMonoidHom.eq_liftOfRightInverse` for the uniqueness lemma.
      ```
         G₁.
         |  \\
       f |   \\ g
         |    \\
         v     \\⌟
         G₂----> G₃
            ∃!φ
      ```"]
def liftOfRightInverse (hf : Function.RightInverse f_inv f) :
    { g : G₁ →* G₃ // f.ker ≤ g.ker } ≃ (G₂ →* G₃) where
  toFun g := f.liftOfRightInverseAux f_inv hf g.1 g.2
                                                      /-
                                                        G : Type u_1
                                                        G' : Type u_2
                                                        G'' : Type u_3
                                                        inst✝⁶ : Group G
                                                        inst✝⁵ : Group G'
                                                        inst✝⁴ : Group G''
                                                        A : Type u_4
                                                        inst✝³ : AddGroup A
                                                        G₁ : Type u_5
                                                        G₂ : Type u_6
                                                        G₃ : Type u_7
                                                        inst✝² : Group G₁
                                                        inst✝¹ : Group G₂
                                                        inst✝ : Group G₃
                                                        f : MonoidHom G₁ G₂
                                                        f_inv : G₂ → G₁
                                                        hf : Function.RightInverse f_inv ⇑f
                                                        φ : MonoidHom G₂ G₃
                                                        x : G₁
                                                        hx : Membership.mem f.ker x
                                                        ⊢ Eq ((φ.comp f) x) 1
                                                      -/
  invFun φ := ⟨φ.comp f, fun x hx ↦ mem_ker.mpr <| by simp [mem_ker.mp hx]⟩
                                                      /-
                                                        🎉 no goals
                                                      -/
  left_inv g := by
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : Subtype fun g => LE.le f.ker g.ker
      ⊢ Eq ((fun φ => ⟨φ.comp f, ⋯⟩) ((fun g => f.liftOfRightInverseAux f_inv hf ↑g  …
    -/
    ext
    /-
      case a.h
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      g : Subtype fun g => LE.le f.ker g.ker
      x✝ : G₁
      ⊢ Eq (↑((fun φ => ⟨φ.comp f, ⋯⟩) ((fun g => f.liftOfRightInverseAux f_inv hf ↑ …
    -/
    simp only [comp_apply, liftOfRightInverseAux_comp_apply, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
  right_inv φ := by
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      φ : MonoidHom G₂ G₃
      ⊢ Eq ((fun g => f.liftOfRightInverseAux f_inv hf ↑g ⋯) ((fun φ => ⟨φ.comp f, ⋯ …
    -/
    ext b
    /-
      case h
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      G₁ : Type u_5
      G₂ : Type u_6
      G₃ : Type u_7
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : Group G₃
      f : MonoidHom G₁ G₂
      f_inv : G₂ → G₁
      hf : Function.RightInverse f_inv ⇑f
      φ : MonoidHom G₂ G₃
      b : G₂
      ⊢ Eq (((fun g => f.liftOfRightInverseAux f_inv hf ↑g ⋯) ((fun φ => ⟨φ.comp f,  …
    -/
    simp [liftOfRightInverseAux, hf b]
    /-
      🎉 no goals
    -/


/-- A non-computable version of `MonoidHom.liftOfRightInverse` for when no computable right
inverse is available, that uses `Function.surjInv`. -/
@[to_additive (attr := simp)
      "A non-computable version of `AddMonoidHom.liftOfRightInverse` for when no
      computable right inverse is available."]
noncomputable abbrev liftOfSurjective (hf : Function.Surjective f) :
    { g : G₁ →* G₃ // f.ker ≤ g.ker } ≃ (G₂ →* G₃) :=
  f.liftOfRightInverse (Function.surjInv hf) (Function.rightInverse_surjInv hf)


@[to_additive (attr := simp)]
theorem liftOfRightInverse_comp_apply (hf : Function.RightInverse f_inv f)
    (g : { g : G₁ →* G₃ // f.ker ≤ g.ker }) (x : G₁) :
    (f.liftOfRightInverse f_inv hf g) (f x) = g.1 x :=
  f.liftOfRightInverseAux_comp_apply f_inv hf g.1 g.2 x


@[to_additive (attr := simp)]
theorem liftOfRightInverse_comp (hf : Function.RightInverse f_inv f)
    (g : { g : G₁ →* G₃ // f.ker ≤ g.ker }) : (f.liftOfRightInverse f_inv hf g).comp f = g :=
  MonoidHom.ext <| f.liftOfRightInverse_comp_apply f_inv hf g


@[to_additive]
theorem eq_liftOfRightInverse (hf : Function.RightInverse f_inv f) (g : G₁ →* G₃)
    (hg : f.ker ≤ g.ker) (h : G₂ →* G₃) (hh : h.comp f = g) :
    h = f.liftOfRightInverse f_inv hf ⟨g, hg⟩ := by
  /-
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    h : MonoidHom G₂ G₃
    hh : Eq (h.comp f) g
    ⊢ Eq h ((f.liftOfRightInverse f_inv hf) ⟨g, hg⟩)
  -/
  simp_rw [← hh]
  /-
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝² : Group G₁
    inst✝¹ : Group G₂
    inst✝ : Group G₃
    f : MonoidHom G₁ G₂
    f_inv : G₂ → G₁
    hf : Function.RightInverse f_inv ⇑f
    g : MonoidHom G₁ G₃
    hg : LE.le f.ker g.ker
    h : MonoidHom G₂ G₃
    hh : Eq (h.comp f) g
    ⊢ Eq h ((f.liftOfRightInverse f_inv hf) ⟨h.comp f, ⋯⟩)
  -/
  exact ((f.liftOfRightInverse f_inv hf).apply_symm_apply _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Normal.comap {H : Subgroup N} (hH : H.Normal) (f : G →* N) : (H.comap f).Normal :=
               /-
                 G : Type u_1
                 inst✝¹ : Group G
                 N : Type u_5
                 inst✝ : Group N
                 H : Subgroup N
                 hH : H.Normal
                 f : MonoidHom G N
                 x✝ : G
                 ⊢ Membership.mem (Subgroup.comap f H) x✝ → ∀ (g : G), Membership.mem (Subgroup …
               -/
  ⟨fun _ => by simp +contextual [Subgroup.mem_comap, hH.conj_mem]⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
instance (priority := 100) normal_comap {H : Subgroup N} [nH : H.Normal] (f : G →* N) :
    (H.comap f).Normal :=
  nH.comap _

-- Here `H.Normal` is an explicit argument so we can use dot notation with `subgroupOf`.

@[to_additive]
theorem Normal.subgroupOf {H : Subgroup G} (hH : H.Normal) (K : Subgroup G) :
    (H.subgroupOf K).Normal :=
  hH.comap _


@[to_additive]
instance (priority := 100) normal_subgroupOf {H N : Subgroup G} [N.Normal] :
    (N.subgroupOf H).Normal :=
  Subgroup.normal_comap _


theorem map_normalClosure (s : Set G) (f : G →* N) (hf : Surjective f) :
    (normalClosure s).map f = normalClosure (f '' s) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    s : Set G
    f : MonoidHom G N
    hf : Function.Surjective ⇑f
    ⊢ Eq (Subgroup.map f (Subgroup.normalClosure s)) (Subgroup.normalClosure (Set. …
  -/
  have : Normal (map f (normalClosure s)) := Normal.map inferInstance f hf
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    s : Set G
    f : MonoidHom G N
    hf : Function.Surjective ⇑f
    this : (Subgroup.map f (Subgroup.normalClosure s)).Normal
    ⊢ Eq (Subgroup.map f (Subgroup.normalClosure s)) (Subgroup.normalClosure (Set. …
  -/
  apply le_antisymm
  · simp [map_le_iff_le_comap, normalClosure_le_normal, coe_comap,
      ← Set.image_subset_iff, subset_normalClosure]
    /-
      case a
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_5
      inst✝ : Group N
      s : Set G
      f : MonoidHom G N
      hf : Function.Surjective ⇑f
      this : (Subgroup.map f (Subgroup.normalClosure s)).Normal
      ⊢ LE.le (Subgroup.normalClosure (Set.image (⇑f) s)) (Subgroup.map f (Subgroup. …
    -/
  · exact normalClosure_le_normal (Set.image_subset f subset_normalClosure)
    /-
      🎉 no goals
    -/


theorem comap_normalClosure (s : Set N) (f : G ≃* N) :
    normalClosure (f ⁻¹' s) = (normalClosure s).comap f := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    s : Set N
    f : MulEquiv G N
    ⊢ Eq (Subgroup.normalClosure (Set.preimage (⇑f) s)) (Subgroup.comap (↑f) (Subg …
  -/
  have := Set.preimage_equiv_eq_image_symm s f.toEquiv
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    s : Set N
    f : MulEquiv G N
    this : Eq (Set.preimage (⇑f.toEquiv) s) (Set.image (⇑f.symm) s)
    ⊢ Eq (Subgroup.normalClosure (Set.preimage (⇑f) s)) (Subgroup.comap (↑f) (Subg …
  -/
  simp_all [comap_equiv_eq_map_symm, map_normalClosure s (f.symm : N →* G) f.symm.surjective]
  /-
    🎉 no goals
  -/


lemma Normal.of_map_injective {G H : Type*} [Group G] [Group H] {φ : G →* H}
    (hφ : Function.Injective φ) {L : Subgroup G} (n : (L.map φ).Normal) : L.Normal :=
  L.comap_map_eq_self_of_injective hφ ▸ n.comap φ


theorem Normal.of_map_subtype {K : Subgroup G} {L : Subgroup K}
    (n : (Subgroup.map K.subtype L).Normal) : L.Normal :=
  n.of_map_injective K.subtype_injective


@[to_additive]
theorem normal_subgroupOf_iff {H K : Subgroup G} (hHK : H ≤ K) :
    (H.subgroupOf K).Normal ↔ ∀ h k, h ∈ H → k ∈ K → k * h * k⁻¹ ∈ H :=
  ⟨fun hN h k hH hK => hN.conj_mem ⟨h, hHK hH⟩ hH ⟨k, hK⟩, fun hN =>
    { conj_mem := fun h hm k => hN h.1 k.1 hm k.2 }⟩


@[to_additive]
instance prod_subgroupOf_prod_normal {H₁ K₁ : Subgroup G} {H₂ K₂ : Subgroup N}
    [h₁ : (H₁.subgroupOf K₁).Normal] [h₂ : (H₂.subgroupOf K₂).Normal] :
    ((H₁.prod H₂).subgroupOf (K₁.prod K₂)).Normal where
  conj_mem n hgHK g :=
    ⟨h₁.conj_mem ⟨(n : G × N).fst, (mem_prod.mp n.2).1⟩ hgHK.1
        ⟨(g : G × N).fst, (mem_prod.mp g.2).1⟩,
      h₂.conj_mem ⟨(n : G × N).snd, (mem_prod.mp n.2).2⟩ hgHK.2
        ⟨(g : G × N).snd, (mem_prod.mp g.2).2⟩⟩


@[to_additive]
instance prod_normal (H : Subgroup G) (K : Subgroup N) [hH : H.Normal] [hK : K.Normal] :
    (H.prod K).Normal where
  conj_mem n hg g :=
    ⟨hH.conj_mem n.fst (Subgroup.mem_prod.mp hg).1 g.fst,
      hK.conj_mem n.snd (Subgroup.mem_prod.mp hg).2 g.snd⟩


@[to_additive]
theorem inf_subgroupOf_inf_normal_of_right (A B' B : Subgroup G) (hB : B' ≤ B)
    [hN : (B'.subgroupOf B).Normal] : ((A ⊓ B').subgroupOf (A ⊓ B)).Normal :=
  { conj_mem := fun {n} hn g =>
      ⟨mul_mem (mul_mem (mem_inf.1 g.2).1 (mem_inf.1 n.2).1) <|
        show ↑g⁻¹ ∈ A from (inv_mem (mem_inf.1 g.2).1),
        (normal_subgroupOf_iff hB).mp hN n g hn.2 (mem_inf.mp g.2).2⟩ }


@[to_additive]
theorem inf_subgroupOf_inf_normal_of_left {A' A : Subgroup G} (B : Subgroup G) (hA : A' ≤ A)
    [hN : (A'.subgroupOf A).Normal] : ((A' ⊓ B).subgroupOf (A ⊓ B)).Normal :=
  { conj_mem := fun n hn g =>
      ⟨(normal_subgroupOf_iff hA).mp hN n g hn.1 (mem_inf.mp g.2).1,
        mul_mem (mul_mem (mem_inf.1 g.2).2 (mem_inf.1 n.2).2) <|
        show ↑g⁻¹ ∈ B from (inv_mem (mem_inf.1 g.2).2)⟩ }


@[to_additive]
instance normal_inf_normal (H K : Subgroup G) [hH : H.Normal] [hK : K.Normal] : (H ⊓ K).Normal :=
  ⟨fun n hmem g => ⟨hH.conj_mem n hmem.1 g, hK.conj_mem n hmem.2 g⟩⟩


@[to_additive]
theorem SubgroupNormal.mem_comm {H K : Subgroup G} (hK : H ≤ K) [hN : (H.subgroupOf K).Normal]
    {a b : G} (hb : b ∈ K) (h : a * b ∈ H) : b * a ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hK : LE.le H K
    hN : (H.subgroupOf K).Normal
    a b : G
    hb : Membership.mem K b
    h : Membership.mem H (HMul.hMul a b)
    ⊢ Membership.mem H (HMul.hMul b a)
  -/
  have := (normal_subgroupOf_iff hK).mp hN (a * b) b h hb
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hK : LE.le H K
    hN : (H.subgroupOf K).Normal
    a b : G
    hb : Membership.mem K b
    h : Membership.mem H (HMul.hMul a b)
    this : Membership.mem H (HMul.hMul (HMul.hMul b (HMul.hMul a b)) (Inv.inv b))
    ⊢ Membership.mem H (HMul.hMul b a)
  -/
  rwa [mul_assoc, mul_assoc, mul_inv_cancel, mul_one] at this
  /-
    🎉 no goals
  -/


/-- Elements of disjoint, normal subgroups commute. -/
@[to_additive "Elements of disjoint, normal subgroups commute."]
theorem commute_of_normal_of_disjoint (H₁ H₂ : Subgroup G) (hH₁ : H₁.Normal) (hH₂ : H₂.Normal)
    (hdis : Disjoint H₁ H₂) (x y : G) (hx : x ∈ H₁) (hy : y ∈ H₂) : Commute x y := by
  suffices x * y * x⁻¹ * y⁻¹ = 1 by
    show x * y = y * x
    · rw [mul_assoc, mul_eq_one_iff_eq_inv] at this
      -- Porting note: Previous code was:
      -- simpa
      simp only [this, mul_inv_rev, inv_inv]
  /-
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ : Subgroup G
    hH₁ : H₁.Normal
    hH₂ : H₂.Normal
    hdis : Disjoint H₁ H₂
    x y : G
    hx : Membership.mem H₁ x
    hy : Membership.mem H₂ y
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul x y) (Inv.inv x)) (Inv.inv y)) 1
  -/
  apply hdis.le_bot
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ : Subgroup G
    hH₁ : H₁.Normal
    hH₂ : H₂.Normal
    hdis : Disjoint H₁ H₂
    x y : G
    hx : Membership.mem H₁ x
    hy : Membership.mem H₂ y
    ⊢ Membership.mem (Min.min H₁ H₂) (HMul.hMul (HMul.hMul (HMul.hMul x y) (Inv.in …
  -/
  constructor
    /-
      case a.left
      G : Type u_1
      inst✝ : Group G
      H₁ H₂ : Subgroup G
      hH₁ : H₁.Normal
      hH₂ : H₂.Normal
      hdis : Disjoint H₁ H₂
      x y : G
      hx : Membership.mem H₁ x
      hy : Membership.mem H₂ y
      ⊢ Membership.mem (↑H₁.toSubmonoid) (HMul.hMul (HMul.hMul (HMul.hMul x y) (Inv. …
    -/
  · suffices x * (y * x⁻¹ * y⁻¹) ∈ H₁ by simpa [mul_assoc]
    /-
      case a.left
      G : Type u_1
      inst✝ : Group G
      H₁ H₂ : Subgroup G
      hH₁ : H₁.Normal
      hH₂ : H₂.Normal
      hdis : Disjoint H₁ H₂
      x y : G
      hx : Membership.mem H₁ x
      hy : Membership.mem H₂ y
      ⊢ Membership.mem H₁ (HMul.hMul x (HMul.hMul (HMul.hMul y (Inv.inv x)) (Inv.inv …
    -/
    exact H₁.mul_mem hx (hH₁.conj_mem _ (H₁.inv_mem hx) _)
    /-
      🎉 no goals
    -/
    /-
      case a.right
      G : Type u_1
      inst✝ : Group G
      H₁ H₂ : Subgroup G
      hH₁ : H₁.Normal
      hH₂ : H₂.Normal
      hdis : Disjoint H₁ H₂
      x y : G
      hx : Membership.mem H₁ x
      hy : Membership.mem H₂ y
      ⊢ Membership.mem (↑H₂.toSubmonoid) (HMul.hMul (HMul.hMul (HMul.hMul x y) (Inv. …
    -/
  · show x * y * x⁻¹ * y⁻¹ ∈ H₂
    /-
      case a.right
      G : Type u_1
      inst✝ : Group G
      H₁ H₂ : Subgroup G
      hH₁ : H₁.Normal
      hH₂ : H₂.Normal
      hdis : Disjoint H₁ H₂
      x y : G
      hx : Membership.mem H₁ x
      hy : Membership.mem H₂ y
      ⊢ Membership.mem H₂ (HMul.hMul (HMul.hMul (HMul.hMul x y) (Inv.inv x)) (Inv.in …
    -/
    apply H₂.mul_mem _ (H₂.inv_mem hy)
    /-
      G : Type u_1
      inst✝ : Group G
      H₁ H₂ : Subgroup G
      hH₁ : H₁.Normal
      hH₂ : H₂.Normal
      hdis : Disjoint H₁ H₂
      x y : G
      hx : Membership.mem H₁ x
      hy : Membership.mem H₂ y
      ⊢ Membership.mem H₂ (HMul.hMul (HMul.hMul x y) (Inv.inv x))
    -/
    apply hH₂.conj_mem _ hy
    /-
      🎉 no goals
    -/


theorem normalClosure_eq_top_of {N : Subgroup G} [hn : N.Normal] {g g' : G} {hg : g ∈ N}
    {hg' : g' ∈ N} (hc : IsConj g g') (ht : normalClosure ({⟨g, hg⟩} : Set N) = ⊤) :
    normalClosure ({⟨g', hg'⟩} : Set N) = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    hn : N.Normal
    g g' : G
    hg : Membership.mem N g
    hg' : Membership.mem N g'
    hc : IsConj g g'
    ht : Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, hg⟩)) Top.top
    ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g', hg'⟩)) Top.top
  -/
  obtain ⟨c, rfl⟩ := isConj_iff.1 hc
  have h : ∀ x : N, (MulAut.conj c) x ∈ N := by
    rintro ⟨x, hx⟩
    exact hn.conj_mem _ hx c
  have hs : Function.Surjective (((MulAut.conj c).toMonoidHom.restrict N).codRestrict _ h) := by
    rintro ⟨x, hx⟩
    refine ⟨⟨c⁻¹ * x * c, ?_⟩, ?_⟩
    · have h := hn.conj_mem _ hx c⁻¹
      rwa [inv_inv] at h
    simp only [MonoidHom.codRestrict_apply, MulEquiv.coe_toMonoidHom, MulAut.conj_apply, coe_mk,
      MonoidHom.restrict_apply, Subtype.mk_eq_mk, ← mul_assoc, mul_inv_cancel, one_mul]
    rw [mul_assoc, mul_inv_cancel, mul_one]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    hn : N.Normal
    g : G
    hg : Membership.mem N g
    ht : Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, hg⟩)) Top.top
    c : G
    hg' : Membership.mem N (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    hc : IsConj g (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    h : ∀ (x : Subtype fun x => Membership.mem N x), Membership.mem N ((MulAut.con …
    hs : Function.Surjective ⇑(((MulEquiv.toMonoidHom (MulAut.conj c)).restrict N) …
    ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (HMul.hMul c g) ( …
  -/
  rw [eq_top_iff, ← MonoidHom.range_eq_top.2 hs, MonoidHom.range_eq_map]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    hn : N.Normal
    g : G
    hg : Membership.mem N g
    ht : Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, hg⟩)) Top.top
    c : G
    hg' : Membership.mem N (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    hc : IsConj g (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    h : ∀ (x : Subtype fun x => Membership.mem N x), Membership.mem N ((MulAut.con …
    hs : Function.Surjective ⇑(((MulEquiv.toMonoidHom (MulAut.conj c)).restrict N) …
    ⊢ LE.le (Subgroup.map (((MulEquiv.toMonoidHom (MulAut.conj c)).restrict N).cod …
  -/
  refine le_trans (map_mono (eq_top_iff.1 ht)) (map_le_iff_le_comap.2 (normalClosure_le_normal ?_))
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    hn : N.Normal
    g : G
    hg : Membership.mem N g
    ht : Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, hg⟩)) Top.top
    c : G
    hg' : Membership.mem N (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    hc : IsConj g (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    h : ∀ (x : Subtype fun x => Membership.mem N x), Membership.mem N ((MulAut.con …
    hs : Function.Surjective ⇑(((MulEquiv.toMonoidHom (MulAut.conj c)).restrict N) …
    ⊢ HasSubset.Subset (Singleton.singleton ⟨g, hg⟩) ↑(Subgroup.comap (((MulEquiv. …
  -/
  rw [Set.singleton_subset_iff, SetLike.mem_coe]
  simp only [MonoidHom.codRestrict_apply, MulEquiv.coe_toMonoidHom, MulAut.conj_apply, coe_mk,
    MonoidHom.restrict_apply, mem_comap]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    hn : N.Normal
    g : G
    hg : Membership.mem N g
    ht : Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, hg⟩)) Top.top
    c : G
    hg' : Membership.mem N (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    hc : IsConj g (HMul.hMul (HMul.hMul c g) (Inv.inv c))
    h : ∀ (x : Subtype fun x => Membership.mem N x), Membership.mem N ((MulAut.con …
    hs : Function.Surjective ⇑(((MulEquiv.toMonoidHom (MulAut.conj c)).restrict N) …
    ⊢ Membership.mem (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (HMul …
  -/
  exact subset_normalClosure (Set.mem_singleton _)
  /-
    🎉 no goals
  -/


/-- The conjugacy classes that are not trivial. -/
def noncenter (G : Type*) [Monoid G] : Set (ConjClasses G) :=
  {x | x.carrier.Nontrivial}


@[simp] lemma mem_noncenter {G} [Monoid G] (g : ConjClasses G) :
  g ∈ noncenter G ↔ g.carrier.Nontrivial := Iff.rfl



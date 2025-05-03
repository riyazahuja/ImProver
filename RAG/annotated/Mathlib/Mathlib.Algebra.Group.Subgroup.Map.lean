/-- The preimage of a subgroup along a monoid homomorphism is a subgroup. -/
@[to_additive
      "The preimage of an `AddSubgroup` along an `AddMonoid` homomorphism
      is an `AddSubgroup`."]
def comap {N : Type*} [Group N] (f : G →* N) (H : Subgroup N) : Subgroup G :=
  { H.toSubmonoid.comap f with
    carrier := f ⁻¹' H
                                                /-
                                                  G : Type u_1
                                                  G' : Type u_2
                                                  G'' : Type u_3
                                                  inst✝⁶ : Group G
                                                  inst✝⁵ : Group G'
                                                  inst✝⁴ : Group G''
                                                  A : Type u_4
                                                  inst✝³ : AddGroup A
                                                  H✝ K : Subgroup G
                                                  k : Set G
                                                  N✝ : Type u_5
                                                  inst✝² : Group N✝
                                                  P : Type u_6
                                                  inst✝¹ : Group P
                                                  N : Type u_7
                                                  inst✝ : Group N
                                                  f : MonoidHom G N
                                                  H : Subgroup N
                                                  a : G
                                                  ha : Membership.mem { carrier := Set.preimage ⇑f ↑H, mul_mem' := ⋯, one_mem' : …
                                                  ⊢ Membership.mem H (f (Inv.inv a))
                                                -/
    inv_mem' := fun {a} ha => show f a⁻¹ ∈ H by rw [f.map_inv]; exact H.inv_mem ha }
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive (attr := simp)]
theorem coe_comap (K : Subgroup N) (f : G →* N) : (K.comap f : Set G) = f ⁻¹' K :=
  rfl


@[to_additive (attr := simp)]
theorem mem_comap {K : Subgroup N} {f : G →* N} {x : G} : x ∈ K.comap f ↔ f x ∈ K :=
  Iff.rfl


@[to_additive]
theorem comap_mono {f : G →* N} {K K' : Subgroup N} : K ≤ K' → comap f K ≤ comap f K' :=
  preimage_mono


@[to_additive]
theorem comap_comap (K : Subgroup P) (g : N →* P) (f : G →* N) :
    (K.comap g).comap f = K.comap (g.comp f) :=
  rfl


@[to_additive (attr := simp)]
theorem comap_id (K : Subgroup N) : K.comap (MonoidHom.id _) = K := by
  /-
    N : Type u_5
    inst✝ : Group N
    K : Subgroup N
    ⊢ Eq (Subgroup.comap (MonoidHom.id N) K) K
  -/
  ext
  /-
    case h
    N : Type u_5
    inst✝ : Group N
    K : Subgroup N
    x✝ : N
    ⊢ Iff (Membership.mem (Subgroup.comap (MonoidHom.id N) K) x✝) (Membership.mem  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toAddSubgroup_comap {G₂ : Type*} [Group G₂] (f : G →* G₂) (s : Subgroup G₂) :
    s.toAddSubgroup.comap (MonoidHom.toAdditive f) = Subgroup.toAddSubgroup (s.comap f) := rfl


@[simp]
theorem _root_.AddSubgroup.toSubgroup_comap {A A₂ : Type*} [AddGroup A] [AddGroup A₂]
    (f : A →+ A₂) (s : AddSubgroup A₂) :
    s.toSubgroup.comap (AddMonoidHom.toMultiplicative f) = AddSubgroup.toSubgroup (s.comap f) := rfl


/-- The image of a subgroup along a monoid homomorphism is a subgroup. -/
@[to_additive
      "The image of an `AddSubgroup` along an `AddMonoid` homomorphism
      is an `AddSubgroup`."]
def map (f : G →* N) (H : Subgroup G) : Subgroup N :=
  { H.toSubmonoid.map f with
    carrier := f '' H
    inv_mem' := by
      /-
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        H✝ K : Subgroup G
        k : Set G
        N : Type u_5
        inst✝¹ : Group N
        P : Type u_6
        inst✝ : Group P
        f : MonoidHom G N
        H : Subgroup G
        ⊢ ∀ {x : N}, Membership.mem { carrier := Set.image ⇑f ↑H, mul_mem' := ⋯, one_m …
      -/
      rintro _ ⟨x, hx, rfl⟩
      /-
        case intro.intro
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        H✝ K : Subgroup G
        k : Set G
        N : Type u_5
        inst✝¹ : Group N
        P : Type u_6
        inst✝ : Group P
        f : MonoidHom G N
        H : Subgroup G
        x : G
        hx : Membership.mem (↑H) x
        ⊢ Membership.mem { carrier := Set.image ⇑f ↑H, mul_mem' := ⋯, one_mem' := ⋯ }. …
      -/
      exact ⟨x⁻¹, H.inv_mem hx, f.map_inv x⟩ }
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem coe_map (f : G →* N) (K : Subgroup G) : (K.map f : Set N) = f '' K :=
  rfl


@[to_additive (attr := simp)]
theorem mem_map {f : G →* N} {K : Subgroup G} {y : N} : y ∈ K.map f ↔ ∃ x ∈ K, f x = y := Iff.rfl


@[to_additive]
theorem mem_map_of_mem (f : G →* N) {K : Subgroup G} {x : G} (hx : x ∈ K) : f x ∈ K.map f :=
  mem_image_of_mem f hx


@[to_additive]
theorem apply_coe_mem_map (f : G →* N) (K : Subgroup G) (x : K) : f x ∈ K.map f :=
  mem_map_of_mem f x.prop


@[to_additive]
theorem map_mono {f : G →* N} {K K' : Subgroup G} : K ≤ K' → map f K ≤ map f K' :=
  image_subset _


@[to_additive (attr := simp)]
theorem map_id : K.map (MonoidHom.id G) = K :=
  SetLike.coe_injective <| image_id _


@[to_additive]
theorem map_map (g : N →* P) (f : G →* N) : (K.map f).map g = K.map (g.comp f) :=
  SetLike.coe_injective <| image_image _ _ _


@[to_additive (attr := simp)]
theorem map_one_eq_bot : K.map (1 : G →* N) = ⊥ :=
  eq_bot_iff.mpr <| by
    /-
      G : Type u_1
      inst✝¹ : Group G
      K : Subgroup G
      N : Type u_5
      inst✝ : Group N
      ⊢ LE.le (Subgroup.map 1 K) Bot.bot
    -/
    rintro x ⟨y, _, rfl⟩
    /-
      case intro.intro
      G : Type u_1
      inst✝¹ : Group G
      K : Subgroup G
      N : Type u_5
      inst✝ : Group N
      y : G
      left✝ : Membership.mem (↑K) y
      ⊢ Membership.mem Bot.bot (1 y)
    -/
    simp
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mem_map_equiv {f : G ≃* N} {K : Subgroup G} {x : N} :
    x ∈ K.map f.toMonoidHom ↔ f.symm x ∈ K := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MulEquiv G N
    K : Subgroup G
    x : N
    ⊢ Iff (Membership.mem (Subgroup.map f.toMonoidHom K) x) (Membership.mem K (f.s …
  -/
  erw [@Set.mem_image_equiv _ _ (↑K) f.toEquiv x]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/

-- The simpNF linter says that the LHS can be simplified via `Subgroup.mem_map`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[to_additive (attr := simp 1100, nolint simpNF)]
theorem mem_map_iff_mem {f : G →* N} (hf : Function.Injective f) {K : Subgroup G} {x : G} :
    f x ∈ K.map f ↔ x ∈ K :=
  hf.mem_set_image


@[to_additive]
theorem map_equiv_eq_comap_symm' (f : G ≃* N) (K : Subgroup G) :
    K.map f.toMonoidHom = K.comap f.symm.toMonoidHom :=
  SetLike.coe_injective (f.toEquiv.image_eq_preimage K)


@[to_additive]
theorem map_equiv_eq_comap_symm (f : G ≃* N) (K : Subgroup G) :
    K.map f = K.comap (G := N) f.symm :=
  map_equiv_eq_comap_symm' _ _


@[to_additive]
theorem comap_equiv_eq_map_symm (f : N ≃* G) (K : Subgroup G) :
    K.comap (G := N) f = K.map f.symm :=
  (map_equiv_eq_comap_symm f.symm K).symm


@[to_additive]
theorem comap_equiv_eq_map_symm' (f : N ≃* G) (K : Subgroup G) :
    K.comap f.toMonoidHom = K.map f.symm.toMonoidHom :=
  (map_equiv_eq_comap_symm f.symm K).symm


@[to_additive]
theorem map_symm_eq_iff_map_eq {H : Subgroup N} {e : G ≃* N} :
    H.map ↑e.symm = K ↔ K.map ↑e = H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    K : Subgroup G
    N : Type u_5
    inst✝ : Group N
    H : Subgroup N
    e : MulEquiv G N
    ⊢ Iff (Eq (Subgroup.map (↑e.symm) H) K) (Eq (Subgroup.map (↑e) K) H)
  -/
  constructor <;> rintro rfl
  · rw [map_map, ← MulEquiv.coe_monoidHom_trans, MulEquiv.symm_trans_self,
      MulEquiv.coe_monoidHom_refl, map_id]
  · rw [map_map, ← MulEquiv.coe_monoidHom_trans, MulEquiv.self_trans_symm,
      MulEquiv.coe_monoidHom_refl, map_id]


@[to_additive]
theorem map_le_iff_le_comap {f : G →* N} {K : Subgroup G} {H : Subgroup N} :
    K.map f ≤ H ↔ K ≤ H.comap f :=
  image_subset_iff


@[to_additive]
theorem gc_map_comap (f : G →* N) : GaloisConnection (map f) (comap f) := fun _ _ =>
  map_le_iff_le_comap


@[to_additive]
theorem map_sup (H K : Subgroup G) (f : G →* N) : (H ⊔ K).map f = H.map f ⊔ K.map f :=
  (gc_map_comap f).l_sup


@[to_additive]
theorem map_iSup {ι : Sort*} (f : G →* N) (s : ι → Subgroup G) :
    (iSup s).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


@[to_additive]
theorem map_inf (H K : Subgroup G) (f : G →* N) (hf : Function.Injective f) :
    (H ⊓ K).map f = H.map f ⊓ K.map f := SetLike.coe_injective (Set.image_inter hf)


@[to_additive]
theorem map_iInf {ι : Sort*} [Nonempty ι] (f : G →* N) (hf : Function.Injective f)
    (s : ι → Subgroup G) : (iInf s).map f = ⨅ i, (s i).map f := by
  /-
    G : Type u_1
    inst✝² : Group G
    N : Type u_5
    inst✝¹ : Group N
    ι : Sort u_7
    inst✝ : Nonempty ι
    f : MonoidHom G N
    hf : Function.Injective ⇑f
    s : ι → Subgroup G
    ⊢ Eq (Subgroup.map f (iInf s)) (iInf fun i => Subgroup.map f (s i))
  -/
  apply SetLike.coe_injective
  /-
    case a
    G : Type u_1
    inst✝² : Group G
    N : Type u_5
    inst✝¹ : Group N
    ι : Sort u_7
    inst✝ : Nonempty ι
    f : MonoidHom G N
    hf : Function.Injective ⇑f
    s : ι → Subgroup G
    ⊢ Eq ↑(Subgroup.map f (iInf s)) ↑(iInf fun i => Subgroup.map f (s i))
  -/
  simpa using (Set.injOn_of_injective hf).image_iInter_eq (s := SetLike.coe ∘ s)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comap_sup_comap_le (H K : Subgroup N) (f : G →* N) :
    comap f H ⊔ comap f K ≤ comap f (H ⊔ K) :=
  Monotone.le_map_sup (fun _ _ => comap_mono) H K


@[to_additive]
theorem iSup_comap_le {ι : Sort*} (f : G →* N) (s : ι → Subgroup N) :
    ⨆ i, (s i).comap f ≤ (iSup s).comap f :=
  Monotone.le_map_iSup fun _ _ => comap_mono


@[to_additive]
theorem comap_inf (H K : Subgroup N) (f : G →* N) : (H ⊓ K).comap f = H.comap f ⊓ K.comap f :=
  (gc_map_comap f).u_inf


@[to_additive]
theorem comap_iInf {ι : Sort*} (f : G →* N) (s : ι → Subgroup N) :
    (iInf s).comap f = ⨅ i, (s i).comap f :=
  (gc_map_comap f).u_iInf


@[to_additive]
theorem map_inf_le (H K : Subgroup G) (f : G →* N) : map f (H ⊓ K) ≤ map f H ⊓ map f K :=
  le_inf (map_mono inf_le_left) (map_mono inf_le_right)


@[to_additive]
theorem map_inf_eq (H K : Subgroup G) (f : G →* N) (hf : Function.Injective f) :
    map f (H ⊓ K) = map f H ⊓ map f K := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H K : Subgroup G
    f : MonoidHom G N
    hf : Function.Injective ⇑f
    ⊢ Eq (Subgroup.map f (Min.min H K)) (Min.min (Subgroup.map f H) (Subgroup.map  …
  -/
  rw [← SetLike.coe_set_eq]
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    H K : Subgroup G
    f : MonoidHom G N
    hf : Function.Injective ⇑f
    ⊢ Eq ↑(Subgroup.map f (Min.min H K)) ↑(Min.min (Subgroup.map f H) (Subgroup.ma …
  -/
  simp [Set.image_inter hf]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem map_bot (f : G →* N) : (⊥ : Subgroup G).map f = ⊥ :=
  (gc_map_comap f).l_bot


@[to_additive (attr := simp)]
theorem map_top_of_surjective (f : G →* N) (h : Function.Surjective f) : Subgroup.map f ⊤ = ⊤ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    h : Function.Surjective ⇑f
    ⊢ Eq (Subgroup.map f Top.top) Top.top
  -/
  rw [eq_top_iff]
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    h : Function.Surjective ⇑f
    ⊢ LE.le Top.top (Subgroup.map f Top.top)
  -/
  intro x _
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    h : Function.Surjective ⇑f
    x : N
    a✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Subgroup.map f Top.top) x
  -/
  obtain ⟨y, hy⟩ := h x
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    h : Function.Surjective ⇑f
    x : N
    a✝ : Membership.mem Top.top x
    y : G
    hy : Eq (f y) x
    ⊢ Membership.mem (Subgroup.map f Top.top) x
  -/
  exact ⟨y, trivial, hy⟩
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem comap_top (f : G →* N) : (⊤ : Subgroup N).comap f = ⊤ :=
  (gc_map_comap f).u_top


/-- For any subgroups `H` and `K`, view `H ⊓ K` as a subgroup of `K`. -/
@[to_additive "For any subgroups `H` and `K`, view `H ⊓ K` as a subgroup of `K`."]
def subgroupOf (H K : Subgroup G) : Subgroup K :=
  H.comap K.subtype


/-- If `H ≤ K`, then `H` as a subgroup of `K` is isomorphic to `H`. -/
@[to_additive (attr := simps) "If `H ≤ K`, then `H` as a subgroup of `K` is isomorphic to `H`."]
def subgroupOfEquivOfLe {G : Type*} [Group G] {H K : Subgroup G} (h : H ≤ K) :
    H.subgroupOf K ≃* H where
  toFun g := ⟨g.1, g.2⟩
  invFun g := ⟨⟨g.1, h g.2⟩, g.2⟩
  left_inv _g := Subtype.ext (Subtype.ext rfl)
  right_inv _g := Subtype.ext rfl
  map_mul' _g _h := rfl


@[to_additive (attr := simp)]
theorem comap_subtype (H K : Subgroup G) : H.comap K.subtype = H.subgroupOf K :=
  rfl


@[to_additive (attr := simp)]
theorem comap_inclusion_subgroupOf {K₁ K₂ : Subgroup G} (h : K₁ ≤ K₂) (H : Subgroup G) :
    (H.subgroupOf K₂).comap (inclusion h) = H.subgroupOf K₁ :=
  rfl


@[to_additive]
theorem coe_subgroupOf (H K : Subgroup G) : (H.subgroupOf K : Set K) = K.subtype ⁻¹' H :=
  rfl


@[to_additive]
theorem mem_subgroupOf {H K : Subgroup G} {h : K} : h ∈ H.subgroupOf K ↔ (h : G) ∈ H :=
  Iff.rfl

-- TODO(kmill): use `K ⊓ H` order for RHS to match `Subtype.image_preimage_coe`

@[to_additive (attr := simp)]
theorem subgroupOf_map_subtype (H K : Subgroup G) : (H.subgroupOf K).map K.subtype = H ⊓ K :=
                     /-
                       G : Type u_1
                       inst✝ : Group G
                       H K : Subgroup G
                       ⊢ Eq ↑(Subgroup.map K.subtype (H.subgroupOf K)) ↑(Min.min H K)
                     -/
  SetLike.ext' <| by refine Subtype.image_preimage_coe _ _ |>.trans ?_; apply Set.inter_comm
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[to_additive]
theorem map_subgroupOf_eq_of_le {H K : Subgroup G} (h : H ≤ K) :
    (H.subgroupOf K).map K.subtype = H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : LE.le H K
    ⊢ Eq (Subgroup.map K.subtype (H.subgroupOf K)) H
  -/
  rwa [subgroupOf_map_subtype, inf_eq_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem bot_subgroupOf : (⊥ : Subgroup G).subgroupOf H = ⊥ :=
  Eq.symm (Subgroup.ext fun _g => Subtype.ext_iff)


@[to_additive (attr := simp)]
theorem top_subgroupOf : (⊤ : Subgroup G).subgroupOf H = ⊤ :=
  rfl


@[to_additive]
theorem subgroupOf_bot_eq_bot : H.subgroupOf ⊥ = ⊥ :=
  Subsingleton.elim _ _


@[to_additive]
theorem subgroupOf_bot_eq_top : H.subgroupOf ⊥ = ⊤ :=
  Subsingleton.elim _ _


@[to_additive (attr := simp)]
theorem subgroupOf_self : H.subgroupOf H = ⊤ :=
  top_unique fun g _hg => g.2


@[to_additive (attr := simp)]
theorem subgroupOf_inj {H₁ H₂ K : Subgroup G} :
    H₁.subgroupOf K = H₂.subgroupOf K ↔ H₁ ⊓ K = H₂ ⊓ K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ K : Subgroup G
    ⊢ Iff (Eq (H₁.subgroupOf K) (H₂.subgroupOf K)) (Eq (Min.min H₁ K) (Min.min H₂  …
  -/
  simpa only [SetLike.ext_iff, mem_inf, mem_subgroupOf, and_congr_left_iff] using Subtype.forall
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inf_subgroupOf_right (H K : Subgroup G) : (H ⊓ K).subgroupOf K = H.subgroupOf K :=
  subgroupOf_inj.2 (inf_right_idem _ _)


@[to_additive (attr := simp)]
theorem inf_subgroupOf_left (H K : Subgroup G) : (K ⊓ H).subgroupOf K = H.subgroupOf K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Eq ((Min.min K H).subgroupOf K) (H.subgroupOf K)
  -/
  rw [inf_comm, inf_subgroupOf_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem subgroupOf_eq_bot {H K : Subgroup G} : H.subgroupOf K = ⊥ ↔ Disjoint H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Iff (Eq (H.subgroupOf K) Bot.bot) (Disjoint H K)
  -/
  rw [disjoint_iff, ← bot_subgroupOf, subgroupOf_inj, bot_inf_eq]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem subgroupOf_eq_top {H K : Subgroup G} : H.subgroupOf K = ⊤ ↔ K ≤ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Iff (Eq (H.subgroupOf K) Top.top) (LE.le K H)
  -/
  rw [← top_subgroupOf, subgroupOf_inj, top_inf_eq, inf_eq_right]
  /-
    🎉 no goals
  -/


@[to_additive]
instance map_isCommutative (f : G →* G') [H.IsCommutative] : (H.map f).IsCommutative :=
  ⟨⟨by
      /-
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁶ : Group G
        inst✝⁵ : Group G'
        inst✝⁴ : Group G''
        A : Type u_4
        inst✝³ : AddGroup A
        H✝ K : Subgroup G
        k : Set G
        N : Type u_5
        inst✝² : Group N
        P : Type u_6
        inst✝¹ : Group P
        H : Subgroup G
        f : MonoidHom G G'
        inst✝ : H.IsCommutative
        ⊢ ∀ (a b : Subtype fun x => Membership.mem (Subgroup.map f H) x), Eq (HMul.hMu …
      -/
      rintro ⟨-, a, ha, rfl⟩ ⟨-, b, hb, rfl⟩
      /-
        case mk.intro.intro.mk.intro.intro
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁶ : Group G
        inst✝⁵ : Group G'
        inst✝⁴ : Group G''
        A : Type u_4
        inst✝³ : AddGroup A
        H✝ K : Subgroup G
        k : Set G
        N : Type u_5
        inst✝² : Group N
        P : Type u_6
        inst✝¹ : Group P
        H : Subgroup G
        f : MonoidHom G G'
        inst✝ : H.IsCommutative
        a : G
        ha : Membership.mem (↑H) a
        b : G
        hb : Membership.mem (↑H) b
        ⊢ Eq (HMul.hMul ⟨f a, ⋯⟩ ⟨f b, ⋯⟩) (HMul.hMul ⟨f b, ⋯⟩ ⟨f a, ⋯⟩)
      -/
      rw [Subtype.ext_iff, coe_mul, coe_mul, Subtype.coe_mk, Subtype.coe_mk, ← map_mul, ← map_mul]
      /-
        case mk.intro.intro.mk.intro.intro
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁶ : Group G
        inst✝⁵ : Group G'
        inst✝⁴ : Group G''
        A : Type u_4
        inst✝³ : AddGroup A
        H✝ K : Subgroup G
        k : Set G
        N : Type u_5
        inst✝² : Group N
        P : Type u_6
        inst✝¹ : Group P
        H : Subgroup G
        f : MonoidHom G G'
        inst✝ : H.IsCommutative
        a : G
        ha : Membership.mem (↑H) a
        b : G
        hb : Membership.mem (↑H) b
        ⊢ Eq (f (HMul.hMul a b)) (f (HMul.hMul b a))
      -/
      exact congr_arg f (Subtype.ext_iff.mp (mul_comm (⟨a, ha⟩ : H) ⟨b, hb⟩))⟩⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem comap_injective_isCommutative {f : G' →* G} (hf : Injective f) [H.IsCommutative] :
    (H.comap f).IsCommutative :=
  ⟨⟨fun a b =>
      Subtype.ext
        (by
          /-
            G : Type u_1
            G' : Type u_2
            inst✝² : Group G
            inst✝¹ : Group G'
            H : Subgroup G
            f : MonoidHom G' G
            hf : Function.Injective ⇑f
            inst✝ : H.IsCommutative
            a b : Subtype fun x => Membership.mem (Subgroup.comap f H) x
            ⊢ Eq ↑(HMul.hMul a b) ↑(HMul.hMul b a)
          -/
          have := mul_comm (⟨f a, a.2⟩ : H) (⟨f b, b.2⟩ : H)
          rwa [Subtype.ext_iff, coe_mul, coe_mul, coe_mk, coe_mk, ← map_mul, ← map_mul,
            hf.eq_iff] at this)⟩⟩


@[to_additive]
instance subgroupOf_isCommutative [H.IsCommutative] : (H.subgroupOf K).IsCommutative :=
  H.comap_injective_isCommutative Subtype.coe_injective


/--
An isomorphism of groups gives an order isomorphism between the lattices of subgroups,
defined by sending subgroups to their inverse images.

See also `MulEquiv.mapSubgroup` which maps subgroups to their forward images.
-/
@[simps]
def comapSubgroup (f : G ≃* H) : Subgroup H ≃o Subgroup G where
  toFun := Subgroup.comap f
  invFun := Subgroup.comap f.symm
                    /-
                      G : Type u_1
                      G' : Type u_2
                      G'' : Type u_3
                      inst✝⁴ : Group G
                      inst✝³ : Group G'
                      inst✝² : Group G''
                      A : Type u_4
                      inst✝¹ : AddGroup A
                      H : Type u_5
                      inst✝ : Group H
                      f : MulEquiv G H
                      sg : Subgroup H
                      ⊢ Eq (Subgroup.comap (↑f.symm) (Subgroup.comap (↑f) sg)) sg
                    -/
  left_inv sg := by simp [Subgroup.comap_comap]
                    /-
                      🎉 no goals
                    -/
                     /-
                       G : Type u_1
                       G' : Type u_2
                       G'' : Type u_3
                       inst✝⁴ : Group G
                       inst✝³ : Group G'
                       inst✝² : Group G''
                       A : Type u_4
                       inst✝¹ : AddGroup A
                       H : Type u_5
                       inst✝ : Group H
                       f : MulEquiv G H
                       sh : Subgroup G
                       ⊢ Eq (Subgroup.comap (↑f) (Subgroup.comap (↑f.symm) sh)) sh
                     -/
  right_inv sh := by simp [Subgroup.comap_comap]
                     /-
                       🎉 no goals
                     -/
  map_rel_iff' {sg1 sg2} :=
    ⟨fun h => by simpa [Subgroup.comap_comap] using
      Subgroup.comap_mono (f := (f.symm : H →* G)) h, Subgroup.comap_mono⟩


/--
An isomorphism of groups gives an order isomorphism between the lattices of subgroups,
defined by sending subgroups to their forward images.

See also `MulEquiv.comapSubgroup` which maps subgroups to their inverse images.
-/
@[simps]
def mapSubgroup {H : Type*} [Group H] (f : G ≃* H) : Subgroup G ≃o Subgroup H where
  toFun := Subgroup.map f
  invFun := Subgroup.map f.symm
                    /-
                      G : Type u_1
                      G' : Type u_2
                      G'' : Type u_3
                      inst✝⁵ : Group G
                      inst✝⁴ : Group G'
                      inst✝³ : Group G''
                      A : Type u_4
                      inst✝² : AddGroup A
                      H✝ : Type u_5
                      inst✝¹ : Group H✝
                      H : Type u_6
                      inst✝ : Group H
                      f : MulEquiv G H
                      sg : Subgroup G
                      ⊢ Eq (Subgroup.map (↑f.symm) (Subgroup.map (↑f) sg)) sg
                    -/
  left_inv sg := by simp [Subgroup.map_map]
                    /-
                      🎉 no goals
                    -/
                     /-
                       G : Type u_1
                       G' : Type u_2
                       G'' : Type u_3
                       inst✝⁵ : Group G
                       inst✝⁴ : Group G'
                       inst✝³ : Group G''
                       A : Type u_4
                       inst✝² : AddGroup A
                       H✝ : Type u_5
                       inst✝¹ : Group H✝
                       H : Type u_6
                       inst✝ : Group H
                       f : MulEquiv G H
                       sh : Subgroup H
                       ⊢ Eq (Subgroup.map (↑f) (Subgroup.map (↑f.symm) sh)) sh
                     -/
  right_inv sh := by simp [Subgroup.map_map]
                     /-
                       🎉 no goals
                     -/
  map_rel_iff' {sg1 sg2} :=
    ⟨fun h => by simpa [Subgroup.map_map] using
      Subgroup.map_mono (f := (f.symm : H →* G)) h, Subgroup.map_mono⟩


@[to_additive]
theorem map_comap_le (H : Subgroup N) : map f (comap f H) ≤ H :=
  (gc_map_comap f).l_u_le _


@[to_additive]
theorem le_comap_map (H : Subgroup G) : H ≤ comap f (map f H) :=
  (gc_map_comap f).le_u_l _


@[to_additive]
theorem map_eq_comap_of_inverse {f : G →* N} {g : N →* G} (hl : Function.LeftInverse g f)
    (hr : Function.RightInverse g f) (H : Subgroup G) : map f H = comap g H :=
                     /-
                       G : Type u_1
                       inst✝¹ : Group G
                       N : Type u_5
                       inst✝ : Group N
                       f : MonoidHom G N
                       g : MonoidHom N G
                       hl : Function.LeftInverse ⇑g ⇑f
                       hr : Function.RightInverse ⇑g ⇑f
                       H : Subgroup G
                       ⊢ Eq ↑(Subgroup.map f H) ↑(Subgroup.comap g H)
                     -/
  SetLike.ext' <| by rw [coe_map, coe_comap, Set.image_eq_preimage_of_inverse hl hr]
                     /-
                       🎉 no goals
                     -/


/-- A subgroup is isomorphic to its image under an injective function. If you have an isomorphism,
use `MulEquiv.subgroupMap` for better definitional equalities. -/
@[to_additive
      "An additive subgroup is isomorphic to its image under an injective function. If you
      have an isomorphism, use `AddEquiv.addSubgroupMap` for better definitional equalities."]
noncomputable def equivMapOfInjective (H : Subgroup G) (f : G →* N) (hf : Function.Injective f) :
    H ≃* H.map f :=
  { Equiv.Set.image f H hf with map_mul' := fun _ _ => Subtype.ext (f.map_mul _ _) }


@[to_additive (attr := simp)]
theorem coe_equivMapOfInjective_apply (H : Subgroup G) (f : G →* N) (hf : Function.Injective f)
    (h : H) : (equivMapOfInjective H f hf h : N) = f h :=
  rfl


/-- The `MonoidHom` from the preimage of a subgroup to itself. -/
@[to_additive (attr := simps!) "the `AddMonoidHom` from the preimage of an
additive subgroup to itself."]
def subgroupComap (f : G →* G') (H' : Subgroup G') : H'.comap f →* H' :=
  f.submonoidComap H'.toSubmonoid


/-- The `MonoidHom` from a subgroup to its image. -/
@[to_additive (attr := simps!) "the `AddMonoidHom` from an additive subgroup to its image"]
def subgroupMap (f : G →* G') (H : Subgroup G) : H →* H.map f :=
  f.submonoidMap H.toSubmonoid


@[to_additive]
theorem subgroupMap_surjective (f : G →* G') (H : Subgroup G) :
    Function.Surjective (f.subgroupMap H) :=
  f.submonoidMap_surjective H.toSubmonoid


/-- Makes the identity isomorphism from a proof two subgroups of a multiplicative
    group are equal. -/
@[to_additive
      "Makes the identity additive isomorphism from a proof
      two subgroups of an additive group are equal."]
def subgroupCongr (h : H = K) : H ≃* K :=
  { Equiv.setCongr <| congr_arg _ h with map_mul' := fun _ _ => rfl }


@[to_additive (attr := simp)]
lemma subgroupCongr_apply (h : H = K) (x) :
    (MulEquiv.subgroupCongr h x : G) = x := rfl


@[to_additive (attr := simp)]
lemma subgroupCongr_symm_apply (h : H = K) (x) :
    ((MulEquiv.subgroupCongr h).symm x : G) = x := rfl


/-- A subgroup is isomorphic to its image under an isomorphism. If you only have an injective map,
use `Subgroup.equiv_map_of_injective`. -/
@[to_additive
      "An additive subgroup is isomorphic to its image under an isomorphism. If you only
      have an injective map, use `AddSubgroup.equiv_map_of_injective`."]
def subgroupMap (e : G ≃* G') (H : Subgroup G) : H ≃* H.map (e : G →* G') :=
  MulEquiv.submonoidMap (e : G ≃* G') H.toSubmonoid


@[to_additive (attr := simp)]
theorem coe_subgroupMap_apply (e : G ≃* G') (H : Subgroup G) (g : H) :
    ((subgroupMap e H g : H.map (e : G →* G')) : G') = e g :=
  rfl


@[to_additive (attr := simp)]
theorem subgroupMap_symm_apply (e : G ≃* G') (H : Subgroup G) (g : H.map (e : G →* G')) :
    (e.subgroupMap H).symm g = ⟨e.symm g, SetLike.mem_coe.1 <| Set.mem_image_equiv.1 g.2⟩ :=
  rfl


@[to_additive]
theorem closure_preimage_le (f : G →* N) (s : Set N) : closure (f ⁻¹' s) ≤ (closure s).comap f :=
                                  /-
                                    G : Type u_1
                                    inst✝¹ : Group G
                                    N : Type u_5
                                    inst✝ : Group N
                                    f : MonoidHom G N
                                    s : Set N
                                    x : G
                                    hx : Membership.mem (Set.preimage (⇑f) s) x
                                    ⊢ Membership.mem (↑(Subgroup.comap f (Subgroup.closure s))) x
                                  -/
  (closure_le _).2 fun x hx => by rw [SetLike.mem_coe, mem_comap]; exact subset_closure hx
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The image under a monoid homomorphism of the subgroup generated by a set equals the subgroup
generated by the image of the set. -/
@[to_additive
      "The image under an `AddMonoid` hom of the `AddSubgroup` generated by a set equals
      the `AddSubgroup` generated by the image of the set."]
theorem map_closure (f : G →* N) (s : Set G) : (closure s).map f = closure (f '' s) :=
  Set.image_preimage.l_comm_of_u_comm (gc_map_comap f) (Subgroup.gi N).gc (Subgroup.gi G).gc
    fun _ ↦ rfl


@[to_additive (attr := simp)]
theorem equivMapOfInjective_coe_mulEquiv (H : Subgroup G) (e : G ≃* G') :
    H.equivMapOfInjective (e : G →* G') (EquivLike.injective e) = e.subgroupMap H := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    e : MulEquiv G G'
    ⊢ Eq (H.equivMapOfInjective ↑e ⋯) (e.subgroupMap H)
  -/
  ext
  /-
    case h.a
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    e : MulEquiv G G'
    x✝ : Subtype fun x => Membership.mem H x
    ⊢ Eq ↑((H.equivMapOfInjective ↑e ⋯) x✝) ↑((e.subgroupMap H) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/



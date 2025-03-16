/-- The index of a subgroup as a natural number. Returns `0` if the index is infinite. -/
@[to_additive "The index of an additive subgroup as a natural number.
Returns 0 if the index is infinite."]
noncomputable def index : ℕ :=
  Nat.card (G ⧸ H)


/-- If `H` and `K` are subgroups of a group `G`, then `relindex H K : ℕ` is the index
of `H ∩ K` in `K`. The function returns `0` if the index is infinite. -/
@[to_additive "If `H` and `K` are subgroups of an additive group `G`, then `relindex H K : ℕ`
is the index of `H ∩ K` in `K`. The function returns `0` if the index is infinite."]
noncomputable def relindex : ℕ :=
  (H.subgroupOf K).index


@[to_additive]
theorem index_comap_of_surjective {f : G' →* G} (hf : Function.Surjective f) :
    (H.comap f).index = H.index := by
  have key : ∀ x y : G',
      QuotientGroup.leftRel (H.comap f) x y ↔ QuotientGroup.leftRel H (f x) (f y) := by
    simp only [QuotientGroup.leftRel_apply]
    exact fun x y => iff_of_eq (congr_arg (· ∈ H) (by rw [f.map_mul, f.map_inv]))
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G' G
    hf : Function.Surjective ⇑f
    key : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) ((Q …
    ⊢ Eq (Subgroup.comap f H).index H.index
  -/
  refine Cardinal.toNat_congr (Equiv.ofBijective (Quotient.map' f fun x y => (key x y).mp) ⟨?_, ?_⟩)
    /-
      case refine_1
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) ((Q …
      ⊢ Function.Injective (Quotient.map' ⇑f ⋯)
    -/
  · simp_rw [← Quotient.eq''] at key
    /-
      case refine_1
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key✝ : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) (( …
      key : ∀ (x y : G'), Iff (Eq (Quotient.mk'' x) (Quotient.mk'' y)) (Eq (Quotient …
      ⊢ Function.Injective (Quotient.map' ⇑f ⋯)
    -/
    refine Quotient.ind' fun x => ?_
    /-
      case refine_1
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key✝ : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) (( …
      key : ∀ (x y : G'), Iff (Eq (Quotient.mk'' x) (Quotient.mk'' y)) (Eq (Quotient …
      x : G'
      ⊢ ∀ ⦃a₂ : HasQuotient.Quotient G' (Subgroup.comap f H)⦄, Eq (Quotient.map' ⇑f  …
    -/
    refine Quotient.ind' fun y => ?_
    /-
      case refine_1
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key✝ : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) (( …
      key : ∀ (x y : G'), Iff (Eq (Quotient.mk'' x) (Quotient.mk'' y)) (Eq (Quotient …
      x y : G'
      ⊢ Eq (Quotient.map' ⇑f ⋯ (Quotient.mk'' x)) (Quotient.map' ⇑f ⋯ (Quotient.mk'' …
    -/
    exact (key x y).mpr
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) ((Q …
      ⊢ Function.Surjective (Quotient.map' ⇑f ⋯)
    -/
  · refine Quotient.ind' fun x => ?_
    /-
      case refine_2
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) ((Q …
      x : G
      ⊢ Exists fun a => Eq (Quotient.map' ⇑f ⋯ a) (Quotient.mk'' x)
    -/
    obtain ⟨y, hy⟩ := hf x
    /-
      case refine_2.intro
      G : Type u_1
      G' : Type u_2
      inst✝¹ : Group G
      inst✝ : Group G'
      H : Subgroup G
      f : MonoidHom G' G
      hf : Function.Surjective ⇑f
      key : ∀ (x y : G'), Iff ((QuotientGroup.leftRel (Subgroup.comap f H)) x y) ((Q …
      x : G
      y : G'
      hy : Eq (f y) x
      ⊢ Exists fun a => Eq (Quotient.map' ⇑f ⋯ a) (Quotient.mk'' x)
    -/
    exact ⟨y, (Quotient.map'_mk'' f _ y).trans (congr_arg Quotient.mk'' hy)⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem index_comap (f : G' →* G) :
    (H.comap f).index = H.relindex f.range :=
                                /-
                                  G : Type u_1
                                  G' : Type u_2
                                  inst✝¹ : Group G
                                  inst✝ : Group G'
                                  H : Subgroup G
                                  f : MonoidHom G' G
                                  ⊢ Eq (Subgroup.comap f H) (Subgroup.comap f.rangeRestrict (H.subgroupOf f.rang …
                                -/
  Eq.trans (congr_arg index (by rfl))
                                /-
                                  🎉 no goals
                                -/
    ((H.subgroupOf f.range).index_comap_of_surjective f.rangeRestrict_surjective)


@[to_additive]
theorem relindex_comap (f : G' →* G) (K : Subgroup G') :
    relindex (comap f H) K = relindex H (map f K) := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G' G
    K : Subgroup G'
    ⊢ Eq ((Subgroup.comap f H).relindex K) (H.relindex (Subgroup.map f K))
  -/
  rw [relindex, subgroupOf, comap_comap, index_comap, ← f.map_range, K.range_subtype]
  /-
    🎉 no goals
  -/


@[to_additive relindex_mul_index]
theorem relindex_mul_index (h : H ≤ K) : H.relindex K * K.index = H.index :=
  ((mul_comm _ _).trans (Cardinal.toNat_mul _ _).symm).trans
    (congr_arg Cardinal.toNat (Equiv.cardinal_eq (quotientEquivProdOfLE h))).symm


@[to_additive]
theorem index_dvd_of_le (h : H ≤ K) : K.index ∣ H.index :=
  dvd_of_mul_left_eq (H.relindex K) (relindex_mul_index h)


@[to_additive]
theorem relindex_dvd_index_of_le (h : H ≤ K) : H.relindex K ∣ H.index :=
  dvd_of_mul_right_eq K.index (relindex_mul_index h)


@[to_additive]
theorem relindex_subgroupOf (hKL : K ≤ L) :
    (H.subgroupOf L).relindex (K.subgroupOf L) = H.relindex K :=
  ((index_comap (H.subgroupOf L) (inclusion hKL)).trans (congr_arg _ (inclusion_range hKL))).symm


@[to_additive relindex_mul_relindex]
theorem relindex_mul_relindex (hHK : H ≤ K) (hKL : K ≤ L) :
    H.relindex K * K.relindex L = H.relindex L := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hHK : LE.le H K
    hKL : LE.le K L
    ⊢ Eq (HMul.hMul (H.relindex K) (K.relindex L)) (H.relindex L)
  -/
  rw [← relindex_subgroupOf hKL]
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hHK : LE.le H K
    hKL : LE.le K L
    ⊢ Eq (HMul.hMul ((H.subgroupOf L).relindex (K.subgroupOf L)) (K.relindex L)) ( …
  -/
  exact relindex_mul_index fun x hx => hHK hx
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inf_relindex_right : (H ⊓ K).relindex K = H.relindex K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Eq ((Min.min H K).relindex K) (H.relindex K)
  -/
  rw [relindex, relindex, inf_subgroupOf_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inf_relindex_left : (H ⊓ K).relindex H = K.relindex H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Eq ((Min.min H K).relindex H) (K.relindex H)
  -/
  rw [inf_comm, inf_relindex_right]
  /-
    🎉 no goals
  -/


@[to_additive relindex_inf_mul_relindex]
theorem relindex_inf_mul_relindex : H.relindex (K ⊓ L) * K.relindex L = (H ⊓ K).relindex L := by
  rw [← inf_relindex_right H (K ⊓ L), ← inf_relindex_right K L, ← inf_relindex_right (H ⊓ K) L,
    inf_assoc, relindex_mul_relindex (H ⊓ (K ⊓ L)) (K ⊓ L) L inf_le_right inf_le_right]


@[to_additive (attr := simp)]
theorem relindex_sup_right [K.Normal] : K.relindex (H ⊔ K) = K.relindex H :=
  Nat.card_congr (QuotientGroup.quotientInfEquivProdNormalQuotient H K).toEquiv.symm


@[to_additive (attr := simp)]
theorem relindex_sup_left [K.Normal] : K.relindex (K ⊔ H) = K.relindex H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    inst✝ : K.Normal
    ⊢ Eq (K.relindex (Max.max K H)) (K.relindex H)
  -/
  rw [sup_comm, relindex_sup_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem relindex_dvd_index_of_normal [H.Normal] : H.relindex K ∣ H.index :=
  relindex_sup_right K H ▸ relindex_dvd_index_of_le le_sup_right


@[to_additive]
theorem relindex_dvd_of_le_left (hHK : H ≤ K) : K.relindex L ∣ H.relindex L :=
  inf_of_le_left hHK ▸ dvd_of_mul_left_eq _ (relindex_inf_mul_relindex _ _ _)


/-- A subgroup has index two if and only if there exists `a` such that for all `b`, exactly one
of `b * a` and `b` belong to `H`. -/
@[to_additive "An additive subgroup has index two if and only if there exists `a` such that
for all `b`, exactly one of `b + a` and `b` belong to `H`."]
theorem index_eq_two_iff : H.index = 2 ↔ ∃ a, ∀ b, Xor' (b * a ∈ H) (b ∈ H) := by
  simp only [index, Nat.card_eq_two_iff' ((1 : G) : G ⧸ H), ExistsUnique, inv_mem_iff,
    QuotientGroup.exists_mk, QuotientGroup.forall_mk, Ne, QuotientGroup.eq, mul_one,
    xor_iff_iff_not]
  refine exists_congr fun a =>
    ⟨fun ha b => ⟨fun hba hb => ?_, fun hb => ?_⟩, fun ha => ⟨?_, fun b hb => ?_⟩⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a : G
      ha : And (Not (Membership.mem H a)) (∀ (x : G), Not (Membership.mem H x) → Mem …
      b : G
      hba : Membership.mem H (HMul.hMul b a)
      hb : Membership.mem H b
      ⊢ False
    -/
  · exact ha.1 ((mul_mem_cancel_left hb).1 hba)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a : G
      ha : And (Not (Membership.mem H a)) (∀ (x : G), Not (Membership.mem H x) → Mem …
      b : G
      hb : Not (Membership.mem H b)
      ⊢ Membership.mem H (HMul.hMul b a)
    -/
  · exact inv_inv b ▸ ha.2 _ (mt (inv_mem_iff (x := b)).1 hb)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a : G
      ha : ∀ (b : G), Iff (Membership.mem H (HMul.hMul b a)) (Not (Membership.mem H  …
      ⊢ Not (Membership.mem H a)
    -/
  · rw [← inv_mem_iff (x := a), ← ha, inv_mul_cancel]
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a : G
      ha : ∀ (b : G), Iff (Membership.mem H (HMul.hMul b a)) (Not (Membership.mem H  …
      ⊢ Membership.mem H 1
    -/
    exact one_mem _
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a : G
      ha : ∀ (b : G), Iff (Membership.mem H (HMul.hMul b a)) (Not (Membership.mem H  …
      b : G
      hb : Not (Membership.mem H b)
      ⊢ Membership.mem H (HMul.hMul (Inv.inv b) a)
    -/
  · rwa [ha, inv_mem_iff (x := b)]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mul_mem_iff_of_index_two (h : H.index = 2) {a b : G} : a * b ∈ H ↔ (a ∈ H ↔ b ∈ H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ⊢ Iff (Membership.mem H (HMul.hMul a b)) (Iff (Membership.mem H a) (Membership …
  -/
  by_cases ha : a ∈ H; · simp only [ha, true_iff, mul_mem_cancel_left ha]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ha : Not (Membership.mem H a)
    ⊢ Iff (Membership.mem H (HMul.hMul a b)) (Iff (Membership.mem H a) (Membership …
  -/
  by_cases hb : b ∈ H; · simp only [hb, iff_true, mul_mem_cancel_right hb]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ha : Not (Membership.mem H a)
    hb : Not (Membership.mem H b)
    ⊢ Iff (Membership.mem H (HMul.hMul a b)) (Iff (Membership.mem H a) (Membership …
  -/
  simp only [ha, hb, iff_true]
  /-
    case neg
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ha : Not (Membership.mem H a)
    hb : Not (Membership.mem H b)
    ⊢ Membership.mem H (HMul.hMul a b)
  -/
  rcases index_eq_two_iff.1 h with ⟨c, hc⟩
  /-
    case neg.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ha : Not (Membership.mem H a)
    hb : Not (Membership.mem H b)
    c : G
    hc : ∀ (b : G), Xor' (Membership.mem H (HMul.hMul b c)) (Membership.mem H b)
    ⊢ Membership.mem H (HMul.hMul a b)
  -/
  refine (hc _).or.resolve_left ?_
  /-
    case neg.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a b : G
    ha : Not (Membership.mem H a)
    hb : Not (Membership.mem H b)
    c : G
    hc : ∀ (b : G), Xor' (Membership.mem H (HMul.hMul b c)) (Membership.mem H b)
    ⊢ Not (Membership.mem H (HMul.hMul (HMul.hMul a b) c))
  -/
  rwa [mul_assoc, mul_mem_cancel_right ((hc _).or.resolve_right hb)]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_self_mem_of_index_two (h : H.index = 2) (a : G) : a * a ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Eq H.index 2
    a : G
    ⊢ Membership.mem H (HMul.hMul a a)
  -/
  rw [mul_mem_iff_of_index_two h]
  /-
    🎉 no goals
  -/


@[to_additive two_smul_mem_of_index_two]
theorem sq_mem_of_index_two (h : H.index = 2) (a : G) : a ^ 2 ∈ H :=
  (pow_two a).symm ▸ mul_self_mem_of_index_two h a


@[to_additive (attr := simp)]
theorem index_top : (⊤ : Subgroup G).index = 1 :=
  Nat.card_eq_one_iff_unique.mpr ⟨QuotientGroup.subsingleton_quotient_top, ⟨1⟩⟩


@[to_additive (attr := simp)]
theorem index_bot : (⊥ : Subgroup G).index = Nat.card G :=
  Cardinal.toNat_congr QuotientGroup.quotientBot.toEquiv


@[deprecated (since := "2024-06-15")] alias index_bot_eq_card := index_bot


@[to_additive (attr := simp)]
theorem relindex_top_left : (⊤ : Subgroup G).relindex H = 1 :=
  index_top


@[to_additive (attr := simp)]
theorem relindex_top_right : H.relindex ⊤ = H.index := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (H.relindex Top.top) H.index
  -/
  rw [← relindex_mul_index (show H ≤ ⊤ from le_top), index_top, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem relindex_bot_left : (⊥ : Subgroup G).relindex H = Nat.card H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (Bot.bot.relindex H) (Nat.card (Subtype fun x => Membership.mem H x))
  -/
  rw [relindex, bot_subgroupOf, index_bot]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-15")] alias relindex_bot_left_eq_card := relindex_bot_left


@[to_additive (attr := simp)]
                                                    /-
                                                      G : Type u_1
                                                      inst✝ : Group G
                                                      H : Subgroup G
                                                      ⊢ Eq (H.relindex Bot.bot) 1
                                                    -/
theorem relindex_bot_right : H.relindex ⊥ = 1 := by rw [relindex, subgroupOf_bot_eq_top, index_top]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive (attr := simp)]
                                               /-
                                                 G : Type u_1
                                                 inst✝ : Group G
                                                 H : Subgroup G
                                                 ⊢ Eq (H.relindex H) 1
                                               -/
theorem relindex_self : H.relindex H = 1 := by rw [relindex, subgroupOf_self, index_top]
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem index_ker (f : G →* G') : f.ker.index = Nat.card f.range := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    ⊢ Eq f.ker.index (Nat.card (Subtype fun x => Membership.mem f.range x))
  -/
  rw [← MonoidHom.comap_bot, index_comap, relindex_bot_left]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem relindex_ker (f : G →* G') : f.ker.relindex K = Nat.card (K.map f) := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    K : Subgroup G
    f : MonoidHom G G'
    ⊢ Eq (f.ker.relindex K) (Nat.card (Subtype fun x => Membership.mem (Subgroup.m …
  -/
  rw [← MonoidHom.comap_bot, relindex_comap, relindex_bot_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) card_mul_index]
theorem card_mul_index : Nat.card H * H.index = Nat.card G := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem H x)) H.index) (Nat …
  -/
  rw [← relindex_bot_left, ← index_bot]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (HMul.hMul (Bot.bot.relindex H) H.index) Bot.bot.index
  -/
  exact relindex_mul_index bot_le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-15")] alias nat_card_dvd_of_injective := card_dvd_of_injective


@[deprecated (since := "2024-06-15")] alias nat_card_dvd_of_le := card_dvd_of_le


@[to_additive]
theorem card_dvd_of_surjective (f : G →* G') (hf : Function.Surjective f) :
    Nat.card G' ∣ Nat.card G := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Nat.card G') (Nat.card G)
  -/
  rw [← Nat.card_congr (QuotientGroup.quotientKerEquivOfSurjective f hf).toEquiv]
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Nat.card (HasQuotient.Quotient G f.ker)) (Nat.card G)
  -/
  exact Dvd.intro_left (Nat.card f.ker) f.ker.card_mul_index
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-15")] alias nat_card_dvd_of_surjective := card_dvd_of_surjective


@[to_additive]
theorem card_range_dvd (f : G →* G') : Nat.card f.range ∣ Nat.card G :=
  card_dvd_of_surjective f.rangeRestrict f.rangeRestrict_surjective


@[to_additive]
theorem card_map_dvd (f : G →* G') : Nat.card (H.map f) ∣ Nat.card H :=
  card_dvd_of_surjective (f.subgroupMap H) (f.subgroupMap_surjective H)


@[to_additive]
theorem index_map (f : G →* G') :
    (H.map f).index = (H ⊔ f.ker).index * f.range.index := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    ⊢ Eq (Subgroup.map f H).index (HMul.hMul (Max.max H f.ker).index f.range.index)
  -/
  rw [← comap_map_eq, index_comap, relindex_mul_index (H.map_le_range f)]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_map_dvd {f : G →* G'} (hf : Function.Surjective f) :
    (H.map f).index ∣ H.index := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Subgroup.map f H).index H.index
  -/
  rw [index_map, f.range_eq_top_of_surjective hf, index_top, mul_one]
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Max.max H f.ker).index H.index
  -/
  exact index_dvd_of_le le_sup_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem dvd_index_map {f : G →* G'} (hf : f.ker ≤ H) :
    H.index ∣ (H.map f).index := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    hf : LE.le f.ker H
    ⊢ Dvd.dvd H.index (Subgroup.map f H).index
  -/
  rw [index_map, sup_of_le_left hf]
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    hf : LE.le f.ker H
    ⊢ Dvd.dvd H.index (HMul.hMul H.index f.range.index)
  -/
  apply dvd_mul_right
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_map_eq {f : G →* G'} (hf1 : Function.Surjective f)
    (hf2 : f.ker ≤ H) : (H.map f).index = H.index :=
  Nat.dvd_antisymm (H.index_map_dvd hf1) (H.dvd_index_map hf2)


@[to_additive]
theorem index_map_of_injective {f : G →* G'} (hf : Function.Injective f) :
    (H.map f).index = H.index * f.range.index := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    f : MonoidHom G G'
    hf : Function.Injective ⇑f
    ⊢ Eq (Subgroup.map f H).index (HMul.hMul H.index f.range.index)
  -/
  rw [H.index_map, f.ker_eq_bot_iff.mpr hf, sup_bot_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_map_subtype {H : Subgroup G} (K : Subgroup H) :
    (K.map H.subtype).index = K.index * H.index := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    K : Subgroup (Subtype fun x => Membership.mem H x)
    ⊢ Eq (Subgroup.map H.subtype K).index (HMul.hMul K.index H.index)
  -/
  rw [K.index_map_of_injective H.subtype_injective, H.range_subtype]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_eq_card : H.index = Nat.card (G ⧸ H) :=
  rfl


@[to_additive index_mul_card]
theorem index_mul_card : H.index * Nat.card H = Nat.card G := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (HMul.hMul H.index (Nat.card (Subtype fun x => Membership.mem H x))) (Nat …
  -/
  rw [mul_comm, card_mul_index]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_dvd_card : H.index ∣ Nat.card G :=
  ⟨Nat.card H, H.index_mul_card.symm⟩


@[to_additive]
theorem relindex_dvd_card : H.relindex K ∣ Nat.card K :=
  (H.subgroupOf K).index_dvd_card


@[to_additive]
theorem relindex_eq_zero_of_le_left (hHK : H ≤ K) (hKL : K.relindex L = 0) : H.relindex L = 0 :=
  eq_zero_of_zero_dvd (hKL ▸ relindex_dvd_of_le_left L hHK)


@[to_additive]
theorem relindex_eq_zero_of_le_right (hKL : K ≤ L) (hHK : H.relindex K = 0) : H.relindex L = 0 :=
  Finite.card_eq_zero_of_embedding (quotientSubgroupOfEmbeddingOfLE H hKL) hHK


@[to_additive]
theorem index_eq_zero_of_relindex_eq_zero (h : H.relindex K = 0) : H.index = 0 :=
  H.relindex_top_right.symm.trans (relindex_eq_zero_of_le_right le_top h)


@[to_additive]
theorem relindex_le_of_le_left (hHK : H ≤ K) (hHL : H.relindex L ≠ 0) :
    K.relindex L ≤ H.relindex L :=
  Nat.le_of_dvd (Nat.pos_of_ne_zero hHL) (relindex_dvd_of_le_left L hHK)


@[to_additive]
theorem relindex_le_of_le_right (hKL : K ≤ L) (hHL : H.relindex L ≠ 0) :
    H.relindex K ≤ H.relindex L :=
  Finite.card_le_of_embedding' (quotientSubgroupOfEmbeddingOfLE H hKL) fun h => (hHL h).elim


@[to_additive]
theorem relindex_ne_zero_trans (hHK : H.relindex K ≠ 0) (hKL : K.relindex L ≠ 0) :
    H.relindex L ≠ 0 := fun h =>
  mul_ne_zero (mt (relindex_eq_zero_of_le_right (show K ⊓ L ≤ K from inf_le_left)) hHK) hKL
    ((relindex_inf_mul_relindex H K L).trans (relindex_eq_zero_of_le_left inf_le_left h))


@[to_additive]
theorem relindex_inf_ne_zero (hH : H.relindex L ≠ 0) (hK : K.relindex L ≠ 0) :
    (H ⊓ K).relindex L ≠ 0 := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hH : Ne (H.relindex L) 0
    hK : Ne (K.relindex L) 0
    ⊢ Ne ((Min.min H K).relindex L) 0
  -/
  replace hH : H.relindex (K ⊓ L) ≠ 0 := mt (relindex_eq_zero_of_le_right inf_le_right) hH
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hK : Ne (K.relindex L) 0
    hH : Ne (H.relindex (Min.min K L)) 0
    ⊢ Ne ((Min.min H K).relindex L) 0
  -/
  rw [← inf_relindex_right] at hH hK ⊢
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hK : Ne ((Min.min K L).relindex L) 0
    hH : Ne ((Min.min H (Min.min K L)).relindex (Min.min K L)) 0
    ⊢ Ne ((Min.min (Min.min H K) L).relindex L) 0
  -/
  rw [inf_assoc]
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    hK : Ne ((Min.min K L).relindex L) 0
    hH : Ne ((Min.min H (Min.min K L)).relindex (Min.min K L)) 0
    ⊢ Ne ((Min.min H (Min.min K L)).relindex L) 0
  -/
  exact relindex_ne_zero_trans hH hK
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_inf_ne_zero (hH : H.index ≠ 0) (hK : K.index ≠ 0) : (H ⊓ K).index ≠ 0 := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hH : Ne H.index 0
    hK : Ne K.index 0
    ⊢ Ne (Min.min H K).index 0
  -/
  rw [← relindex_top_right] at hH hK ⊢
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hH : Ne (H.relindex Top.top) 0
    hK : Ne (K.relindex Top.top) 0
    ⊢ Ne ((Min.min H K).relindex Top.top) 0
  -/
  exact relindex_inf_ne_zero hH hK
  /-
    🎉 no goals
  -/


@[to_additive]
theorem relindex_inf_le : (H ⊓ K).relindex L ≤ H.relindex L * K.relindex L := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    ⊢ LE.le ((Min.min H K).relindex L) (HMul.hMul (H.relindex L) (K.relindex L))
  -/
  by_cases h : H.relindex L = 0
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      H K L : Subgroup G
      h : Eq (H.relindex L) 0
      ⊢ LE.le ((Min.min H K).relindex L) (HMul.hMul (H.relindex L) (K.relindex L))
    -/
  · exact (le_of_eq (relindex_eq_zero_of_le_left inf_le_left h)).trans (zero_le _)
    /-
      🎉 no goals
    -/
  rw [← inf_relindex_right, inf_assoc, ← relindex_mul_relindex _ _ L inf_le_right inf_le_right,
    inf_relindex_right, inf_relindex_right]
  /-
    case neg
    G : Type u_1
    inst✝ : Group G
    H K L : Subgroup G
    h : Not (Eq (H.relindex L) 0)
    ⊢ LE.le (HMul.hMul (H.relindex (Min.min K L)) (K.relindex L)) (HMul.hMul (H.re …
  -/
  exact mul_le_mul_right' (relindex_le_of_le_right inf_le_right h) (K.relindex L)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_inf_le : (H ⊓ K).index ≤ H.index * K.index := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ LE.le (Min.min H K).index (HMul.hMul H.index K.index)
  -/
  simp_rw [← relindex_top_right, relindex_inf_le]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem relindex_iInf_ne_zero {ι : Type*} [_hι : Finite ι] {f : ι → Subgroup G}
    (hf : ∀ i, (f i).relindex L ≠ 0) : (⨅ i, f i).relindex L ≠ 0 :=
  haveI := Fintype.ofFinite ι
  (Finset.prod_ne_zero_iff.mpr fun i _hi => hf i) ∘
    Nat.card_pi.symm.trans ∘
      Finite.card_eq_zero_of_embedding (quotientiInfSubgroupOfEmbedding f L)


@[to_additive]
theorem relindex_iInf_le {ι : Type*} [Fintype ι] (f : ι → Subgroup G) :
    (⨅ i, f i).relindex L ≤ ∏ i, (f i).relindex L :=
  le_of_le_of_eq
    (Finite.card_le_of_embedding' (quotientiInfSubgroupOfEmbedding f L) fun h =>
      let ⟨i, _hi, h⟩ := Finset.prod_eq_zero_iff.mp (Nat.card_pi.symm.trans h)
      relindex_eq_zero_of_le_left (iInf_le f i) h)
    Nat.card_pi


@[to_additive]
theorem index_iInf_ne_zero {ι : Type*} [Finite ι] {f : ι → Subgroup G}
    (hf : ∀ i, (f i).index ≠ 0) : (⨅ i, f i).index ≠ 0 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → Subgroup G
    hf : ∀ (i : ι), Ne (f i).index 0
    ⊢ Ne (iInf fun i => f i).index 0
  -/
  simp_rw [← relindex_top_right] at hf ⊢
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → Subgroup G
    hf : ∀ (i : ι), Ne ((f i).relindex Top.top) 0
    ⊢ Ne ((iInf fun i => f i).relindex Top.top) 0
  -/
  exact relindex_iInf_ne_zero hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem index_iInf_le {ι : Type*} [Fintype ι] (f : ι → Subgroup G) :
                                              /-
                                                G : Type u_1
                                                inst✝¹ : Group G
                                                ι : Type u_3
                                                inst✝ : Fintype ι
                                                f : ι → Subgroup G
                                                ⊢ LE.le (iInf fun i => f i).index (Finset.univ.prod fun i => (f i).index)
                                              -/
    (⨅ i, f i).index ≤ ∏ i, (f i).index := by simp_rw [← relindex_top_right, relindex_iInf_le]
                                              /-
                                                🎉 no goals
                                              -/

-- Porting note: had to replace `Cardinal.toNat_eq_one_iff_unique` with `Nat.card_eq_one_iff_unique`

@[to_additive (attr := simp) index_eq_one]
theorem index_eq_one : H.index = 1 ↔ H = ⊤ :=
  ⟨fun h =>
    QuotientGroup.subgroup_eq_top_of_subsingleton H (Nat.card_eq_one_iff_unique.mp h).1,
    fun h => (congr_arg index h).trans index_top⟩


@[to_additive (attr := simp) relindex_eq_one]
theorem relindex_eq_one : H.relindex K = 1 ↔ K ≤ H :=
  index_eq_one.trans subgroupOf_eq_top


@[to_additive (attr := simp) card_eq_one]
theorem card_eq_one : Nat.card H = 1 ↔ H = ⊥ :=
  H.relindex_bot_left ▸ relindex_eq_one.trans le_bot_iff


@[to_additive]
lemma inf_eq_bot_of_coprime (h : Nat.Coprime (Nat.card H) (Nat.card K)) : H ⊓ K = ⊥ :=
  card_eq_one.1 <| Nat.eq_one_of_dvd_coprimes h
    (card_dvd_of_le inf_le_left) (card_dvd_of_le inf_le_right)


@[deprecated (since := "2024-12-18")]
alias _root_.add_inf_eq_bot_of_coprime := AddSubgroup.inf_eq_bot_of_coprime


@[to_additive]
theorem index_ne_zero_of_finite [hH : Finite (G ⧸ H)] : H.index ≠ 0 := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hH : Finite (HasQuotient.Quotient G H)
    ⊢ Ne H.index 0
  -/
  cases nonempty_fintype (G ⧸ H)
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hH : Finite (HasQuotient.Quotient G H)
    val✝ : Fintype (HasQuotient.Quotient G H)
    ⊢ Ne H.index 0
  -/
  rw [index_eq_card]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    hH : Finite (HasQuotient.Quotient G H)
    val✝ : Fintype (HasQuotient.Quotient G H)
    ⊢ Ne (Nat.card (HasQuotient.Quotient G H)) 0
  -/
  exact Nat.card_pos.ne'
  /-
    🎉 no goals
  -/

-- Porting note: changed due to error with `Cardinal.toNat_apply_of_aleph0_le`

/-- Finite index implies finite quotient. -/
@[to_additive "Finite index implies finite quotient."]
noncomputable def fintypeOfIndexNeZero (hH : H.index ≠ 0) : Fintype (G ⧸ H) :=
  @Fintype.ofFinite _ (Nat.finite_of_card_ne_zero hH)


@[to_additive]
lemma index_eq_zero_iff_infinite : H.index = 0 ↔ Infinite (G ⧸ H) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Eq H.index 0) (Infinite (HasQuotient.Quotient G H))
  -/
  simp [index_eq_card, Nat.card_eq_zero]
  /-
    🎉 no goals
  -/


@[to_additive one_lt_index_of_ne_top]
theorem one_lt_index_of_ne_top [Finite (G ⧸ H)] (hH : H ≠ ⊤) : 1 < H.index :=
  Nat.one_lt_iff_ne_zero_and_ne_one.mpr ⟨index_ne_zero_of_finite, mt index_eq_one.mp hH⟩


@[to_additive]
lemma finite_quotient_of_finite_quotient_of_index_ne_zero {X : Type*} [MulAction G X]
    [Finite <| MulAction.orbitRel.Quotient G X] (hi : H.index ≠ 0) :
    Finite <| MulAction.orbitRel.Quotient H X := by
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    X : Type u_3
    inst✝¹ : MulAction G X
    inst✝ : Finite (MulAction.orbitRel.Quotient G X)
    hi : Ne H.index 0
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) X)
  -/
  have := fintypeOfIndexNeZero hi
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    X : Type u_3
    inst✝¹ : MulAction G X
    inst✝ : Finite (MulAction.orbitRel.Quotient G X)
    hi : Ne H.index 0
    this : Fintype (HasQuotient.Quotient G H)
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) X)
  -/
  exact MulAction.finite_quotient_of_finite_quotient_of_finite_quotient
  /-
    🎉 no goals
  -/


@[to_additive]
lemma finite_quotient_of_pretransitive_of_index_ne_zero {X : Type*} [MulAction G X]
    [MulAction.IsPretransitive G X] (hi : H.index ≠ 0) :
    Finite <| MulAction.orbitRel.Quotient H X := by
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    X : Type u_3
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    hi : Ne H.index 0
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) X)
  -/
  have := (MulAction.pretransitive_iff_subsingleton_quotient G X).1 inferInstance
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    X : Type u_3
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    hi : Ne H.index 0
    this : Subsingleton (MulAction.orbitRel.Quotient G X)
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) X)
  -/
  exact finite_quotient_of_finite_quotient_of_index_ne_zero hi
  /-
    🎉 no goals
  -/


@[to_additive]
lemma exists_pow_mem_of_index_ne_zero (h : H.index ≠ 0) (a : G) :
    ∃ n, 0 < n ∧ n ≤ H.index ∧ a ^ n ∈ H := by
  suffices ∃ n₁ n₂, n₁ < n₂ ∧ n₂ ≤ H.index ∧ ((a ^ n₂ : G) : G ⧸ H) = ((a ^ n₁ : G) : G ⧸ H) by
    rcases this with ⟨n₁, n₂, hlt, hle, he⟩
    refine ⟨n₂ - n₁, by omega, by omega, ?_⟩
    rw [eq_comm, QuotientGroup.eq, ← zpow_natCast, ← zpow_natCast, ← zpow_neg, ← zpow_add,
        add_comm] at he
    rw [← zpow_natCast]
    convert he
    omega
  suffices ∃ n₁ n₂, n₁ ≠ n₂ ∧ n₁ ≤ H.index ∧ n₂ ≤ H.index ∧
      ((a ^ n₂ : G) : G ⧸ H) = ((a ^ n₁ : G) : G ⧸ H) by
    rcases this with ⟨n₁, n₂, hne, hle₁, hle₂, he⟩
    rcases hne.lt_or_lt with hlt | hlt
    · exact ⟨n₁, n₂, hlt, hle₂, he⟩
    · exact ⟨n₂, n₁, hlt, hle₁, he.symm⟩
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    ⊢ Exists fun n₁ => Exists fun n₂ => And (Ne n₁ n₂) (And (LE.le n₁ H.index) (An …
  -/
  by_contra hc
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    hc : Not (Exists fun n₁ => Exists fun n₂ => And (Ne n₁ n₂) (And (LE.le n₁ H.in …
    ⊢ False
  -/
  simp_rw [not_exists] at hc
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    hc : ∀ (x x_1 : Nat), Not (And (Ne x x_1) (And (LE.le x H.index) (And (LE.le x …
    ⊢ False
  -/
  let f : (Set.Icc 0 H.index) → G ⧸ H := fun n ↦ (a ^ (n : ℕ) : G)
  have hf : Function.Injective f := by
    rintro ⟨n₁, h₁, hle₁⟩ ⟨n₂, h₂, hle₂⟩ he
    have hc' := hc n₁ n₂
    dsimp only [f] at he
    simpa [hle₁, hle₂, he] using hc'
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    hc : ∀ (x x_1 : Nat), Not (And (Ne x x_1) (And (LE.le x H.index) (And (LE.le x …
    f : ↑(Set.Icc 0 H.index) → HasQuotient.Quotient G H := fun n => ↑(HPow.hPow a  …
    hf : Function.Injective f
    ⊢ False
  -/
  have := (fintypeOfIndexNeZero h).finite
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    hc : ∀ (x x_1 : Nat), Not (And (Ne x x_1) (And (LE.le x H.index) (And (LE.le x …
    f : ↑(Set.Icc 0 H.index) → HasQuotient.Quotient G H := fun n => ↑(HPow.hPow a  …
    hf : Function.Injective f
    this : Finite (HasQuotient.Quotient G H)
    ⊢ False
  -/
  have hcard := Finite.card_le_of_injective f hf
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    hc : ∀ (x x_1 : Nat), Not (And (Ne x x_1) (And (LE.le x H.index) (And (LE.le x …
    f : ↑(Set.Icc 0 H.index) → HasQuotient.Quotient G H := fun n => ↑(HPow.hPow a  …
    hf : Function.Injective f
    this : Finite (HasQuotient.Quotient G H)
    hcard : LE.le (Nat.card ↑(Set.Icc 0 H.index)) (Nat.card (HasQuotient.Quotient  …
    ⊢ False
  -/
  simp [← index_eq_card] at hcard
  /-
    🎉 no goals
  -/


@[to_additive]
lemma exists_pow_mem_of_relindex_ne_zero (h : H.relindex K ≠ 0) {a : G} (ha : a ∈ K) :
    ∃ n, 0 < n ∧ n ≤ H.relindex K ∧ a ^ n ∈ H ⊓ K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : Ne (H.relindex K) 0
    a : G
    ha : Membership.mem K a
    ⊢ Exists fun n => And (LT.lt 0 n) (And (LE.le n (H.relindex K)) (Membership.me …
  -/
  rcases exists_pow_mem_of_index_ne_zero h ⟨a, ha⟩ with ⟨n, hlt, hle, he⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : Ne (H.relindex K) 0
    a : G
    ha : Membership.mem K a
    n : Nat
    hlt : LT.lt 0 n
    hle : LE.le n (H.subgroupOf K).index
    he : Membership.mem (H.subgroupOf K) (HPow.hPow ⟨a, ha⟩ n)
    ⊢ Exists fun n => And (LT.lt 0 n) (And (LE.le n (H.relindex K)) (Membership.me …
  -/
  refine ⟨n, hlt, hle, ?_⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : Ne (H.relindex K) 0
    a : G
    ha : Membership.mem K a
    n : Nat
    hlt : LT.lt 0 n
    hle : LE.le n (H.subgroupOf K).index
    he : Membership.mem (H.subgroupOf K) (HPow.hPow ⟨a, ha⟩ n)
    ⊢ Membership.mem (Min.min H K) (HPow.hPow a n)
  -/
  simpa [pow_mem ha, mem_subgroupOf] using he
  /-
    🎉 no goals
  -/


@[to_additive]
lemma pow_mem_of_index_ne_zero_of_dvd (h : H.index ≠ 0) (a : G) {n : ℕ}
    (hn : ∀ m, 0 < m → m ≤ H.index → m ∣ n) : a ^ n ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    n : Nat
    hn : ∀ (m : Nat), LT.lt 0 m → LE.le m H.index → Dvd.dvd m n
    ⊢ Membership.mem H (HPow.hPow a n)
  -/
  rcases exists_pow_mem_of_index_ne_zero h a with ⟨m, hlt, hle, he⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    n : Nat
    hn : ∀ (m : Nat), LT.lt 0 m → LE.le m H.index → Dvd.dvd m n
    m : Nat
    hlt : LT.lt 0 m
    hle : LE.le m H.index
    he : Membership.mem H (HPow.hPow a m)
    ⊢ Membership.mem H (HPow.hPow a n)
  -/
  rcases hn m hlt hle with ⟨k, rfl⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    m : Nat
    hlt : LT.lt 0 m
    hle : LE.le m H.index
    he : Membership.mem H (HPow.hPow a m)
    k : Nat
    hn : ∀ (m_1 : Nat), LT.lt 0 m_1 → LE.le m_1 H.index → Dvd.dvd m_1 (HMul.hMul m …
    ⊢ Membership.mem H (HPow.hPow a (HMul.hMul m k))
  -/
  rw [pow_mul]
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    h : Ne H.index 0
    a : G
    m : Nat
    hlt : LT.lt 0 m
    hle : LE.le m H.index
    he : Membership.mem H (HPow.hPow a m)
    k : Nat
    hn : ∀ (m_1 : Nat), LT.lt 0 m_1 → LE.le m_1 H.index → Dvd.dvd m_1 (HMul.hMul m …
    ⊢ Membership.mem H (HPow.hPow (HPow.hPow a m) k)
  -/
  exact pow_mem he _
  /-
    🎉 no goals
  -/


@[to_additive]
lemma pow_mem_of_relindex_ne_zero_of_dvd (h : H.relindex K ≠ 0) {a : G} (ha : a ∈ K) {n : ℕ}
    (hn : ∀ m, 0 < m → m ≤ H.relindex K → m ∣ n) : a ^ n ∈ H ⊓ K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : Ne (H.relindex K) 0
    a : G
    ha : Membership.mem K a
    n : Nat
    hn : ∀ (m : Nat), LT.lt 0 m → LE.le m (H.relindex K) → Dvd.dvd m n
    ⊢ Membership.mem (Min.min H K) (HPow.hPow a n)
  -/
  convert pow_mem_of_index_ne_zero_of_dvd h ⟨a, ha⟩ hn
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : Ne (H.relindex K) 0
    a : G
    ha : Membership.mem K a
    n : Nat
    hn : ∀ (m : Nat), LT.lt 0 m → LE.le m (H.relindex K) → Dvd.dvd m n
    ⊢ Iff (Membership.mem (Min.min H K) (HPow.hPow a n)) (Membership.mem (H.subgro …
  -/
  simp [pow_mem ha, mem_subgroupOf]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma index_prod (H : Subgroup G) (K : Subgroup G') : (H.prod K).index = H.index * K.index := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    K : Subgroup G'
    ⊢ Eq (H.prod K).index (HMul.hMul H.index K.index)
  -/
  simp_rw [index, ← Nat.card_prod]
  refine Nat.card_congr
    ((Quotient.congrRight (fun x y ↦ ?_)).trans (Setoid.prodQuotientEquiv _ _).symm)
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    H : Subgroup G
    K : Subgroup G'
    x y : Prod G G'
    ⊢ Iff ((QuotientGroup.leftRel (H.prod K)) x y) (((QuotientGroup.leftRel H).pro …
  -/
  rw [QuotientGroup.leftRel_prod]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma index_pi {ι : Type*} [Fintype ι] (H : ι → Subgroup G) :
    (Subgroup.pi Set.univ H).index = ∏ i, (H i).index := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_3
    inst✝ : Fintype ι
    H : ι → Subgroup G
    ⊢ Eq (Subgroup.pi Set.univ H).index (Finset.univ.prod fun i => (H i).index)
  -/
  simp_rw [index, ← Nat.card_pi]
  refine Nat.card_congr
    ((Quotient.congrRight (fun x y ↦ ?_)).trans (Setoid.piQuotientEquiv _).symm)
  /-
    G : Type u_1
    inst✝¹ : Group G
    ι : Type u_3
    inst✝ : Fintype ι
    H : ι → Subgroup G
    x y : ι → G
    ⊢ Iff ((QuotientGroup.leftRel (Subgroup.pi Set.univ H)) x y) (piSetoid x y)
  -/
  rw [QuotientGroup.leftRel_pi]
  /-
    🎉 no goals
  -/


@[simp]
lemma index_toAddSubgroup : (Subgroup.toAddSubgroup H).index = H.index :=
  rfl


@[simp]
lemma _root_.AddSubgroup.index_toSubgroup {G : Type*} [AddGroup G] (H : AddSubgroup G) :
    (AddSubgroup.toSubgroup H).index = H.index :=
  rfl


@[simp]
lemma relindex_toAddSubgroup :
    (Subgroup.toAddSubgroup H).relindex (Subgroup.toAddSubgroup K) = H.relindex K :=
  rfl


@[simp]
lemma _root_.AddSubgroup.relindex_toSubgroup {G : Type*} [AddGroup G] (H K : AddSubgroup G) :
    (AddSubgroup.toSubgroup H).relindex (AddSubgroup.toSubgroup K) = H.relindex K :=
  rfl


/-- Typeclass for finite index subgroups. -/
class FiniteIndex : Prop where
  /-- The subgroup has finite index -/
  finiteIndex : H.index ≠ 0


/-- Typeclass for finite index subgroups. -/
class _root_.AddSubgroup.FiniteIndex {G : Type*} [AddGroup G] (H : AddSubgroup G) : Prop where
  /-- The additive subgroup has finite index -/
  finiteIndex : H.index ≠ 0


/-- A finite index subgroup has finite quotient. -/
@[to_additive "A finite index subgroup has finite quotient"]
noncomputable def fintypeQuotientOfFiniteIndex [FiniteIndex H] : Fintype (G ⧸ H) :=
  fintypeOfIndexNeZero FiniteIndex.finiteIndex


@[to_additive]
instance finite_quotient_of_finiteIndex [FiniteIndex H] : Finite (G ⧸ H) :=
  H.fintypeQuotientOfFiniteIndex.finite


@[to_additive]
theorem finiteIndex_of_finite_quotient [Finite (G ⧸ H)] : FiniteIndex H :=
  ⟨index_ne_zero_of_finite⟩

-- Porting note: had to manually provide finite instance for quotient when it should be automatic

@[to_additive]
instance (priority := 100) finiteIndex_of_finite [Finite G] : FiniteIndex H :=
  @finiteIndex_of_finite_quotient _ _ H (Quotient.finite _)


@[to_additive]
instance : FiniteIndex (⊤ : Subgroup G) :=
  ⟨ne_of_eq_of_ne index_top one_ne_zero⟩


@[to_additive]
instance [FiniteIndex H] [FiniteIndex K] : FiniteIndex (H ⊓ K) :=
  ⟨index_inf_ne_zero FiniteIndex.finiteIndex FiniteIndex.finiteIndex⟩


@[to_additive]
theorem finiteIndex_iInf {ι : Type*} [Finite ι] {f : ι → Subgroup G}
    (hf : ∀ i, (f i).FiniteIndex) : (⨅ i, f i).FiniteIndex :=
  ⟨index_iInf_ne_zero fun i => (hf i).finiteIndex⟩


@[to_additive]
theorem finiteIndex_iInf' {ι : Type*} {s : Finset ι}
    (f : ι → Subgroup G) (hs : ∀ i ∈ s, (f i).FiniteIndex) :
    (⨅ i ∈ s, f i).FiniteIndex := by
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Type u_3
    s : Finset ι
    f : ι → Subgroup G
    hs : ∀ (i : ι), Membership.mem s i → (f i).FiniteIndex
    ⊢ (iInf fun i => iInf fun h => f i).FiniteIndex
  -/
  rw [iInf_subtype']
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Type u_3
    s : Finset ι
    f : ι → Subgroup G
    hs : ∀ (i : ι), Membership.mem s i → (f i).FiniteIndex
    ⊢ (iInf fun x => f ↑x).FiniteIndex
  -/
  exact finiteIndex_iInf fun ⟨i, hi⟩ => hs i hi
  /-
    🎉 no goals
  -/


@[to_additive]
instance instFiniteIndex_subgroupOf (H K : Subgroup G) [H.FiniteIndex] :
    (H.subgroupOf K).FiniteIndex :=
  ⟨fun h => H.index_ne_zero_of_finite <| H.index_eq_zero_of_relindex_eq_zero h⟩


@[to_additive]
theorem finiteIndex_of_le [FiniteIndex H] (h : H ≤ K) : FiniteIndex K :=
  ⟨ne_zero_of_dvd_ne_zero FiniteIndex.finiteIndex (index_dvd_of_le h)⟩


@[to_additive (attr := gcongr)]
lemma index_antitone (h : H ≤ K) [H.FiniteIndex] : K.index ≤ H.index :=
  Nat.le_of_dvd (Nat.zero_lt_of_ne_zero FiniteIndex.finiteIndex) (index_dvd_of_le h)


@[to_additive (attr := gcongr)]
lemma index_strictAnti (h : H < K) [H.FiniteIndex] : K.index < H.index := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    h : LT.lt H K
    inst✝ : H.FiniteIndex
    ⊢ LT.lt K.index H.index
  -/
  have h0 : K.index ≠ 0 := (finiteIndex_of_le h.le).finiteIndex
  /-
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    h : LT.lt H K
    inst✝ : H.FiniteIndex
    h0 : Ne K.index 0
    ⊢ LT.lt K.index H.index
  -/
  apply lt_of_le_of_ne (index_antitone h.le)
  /-
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    h : LT.lt H K
    inst✝ : H.FiniteIndex
    h0 : Ne K.index 0
    ⊢ Ne K.index H.index
  -/
  rw [← relindex_mul_index h.le, Ne, eq_comm, mul_eq_right₀ h0, relindex_eq_one]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    h : LT.lt H K
    inst✝ : H.FiniteIndex
    h0 : Ne K.index 0
    ⊢ Not (LE.le K H)
  -/
  exact h.not_le
  /-
    🎉 no goals
  -/


@[to_additive]
instance finiteIndex_ker {G' : Type*} [Group G'] (f : G →* G') [Finite f.range] :
    f.ker.FiniteIndex :=
  @finiteIndex_of_finite_quotient G _ f.ker
    (Finite.of_equiv f.range (QuotientGroup.quotientKerEquivRange f).symm)


instance finiteIndex_normalCore [H.FiniteIndex] : H.normalCore.FiniteIndex := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    H K L : Subgroup G
    inst✝ : H.FiniteIndex
    ⊢ H.normalCore.FiniteIndex
  -/
  rw [normalCore_eq_ker]
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    H K L : Subgroup G
    inst✝ : H.FiniteIndex
    ⊢ (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker.FiniteIndex
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance finiteIndex_center [Finite (commutatorSet G)] [Group.FG G] : FiniteIndex (center G) := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝³ : Group G
    inst✝² : Group G'
    H K L : Subgroup G
    inst✝¹ : Finite ↑(commutatorSet G)
    inst✝ : Group.FG G
    ⊢ (Subgroup.center G).FiniteIndex
  -/
  obtain ⟨S, -, hS⟩ := Group.rank_spec G
  /-
    case intro.intro
    G : Type u_1
    G' : Type u_2
    inst✝³ : Group G
    inst✝² : Group G'
    H K L : Subgroup G
    inst✝¹ : Finite ↑(commutatorSet G)
    inst✝ : Group.FG G
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ (Subgroup.center G).FiniteIndex
  -/
  exact ⟨mt (Finite.card_eq_zero_of_embedding (quotientCenterEmbedding hS)) Finite.card_pos.ne'⟩
  /-
    🎉 no goals
  -/


theorem index_center_le_pow [Finite (commutatorSet G)] [Group.FG G] :
    (center G).index ≤ Nat.card (commutatorSet G) ^ Group.rank G := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite ↑(commutatorSet G)
    inst✝ : Group.FG G
    ⊢ LE.le (Subgroup.center G).index (HPow.hPow (Nat.card ↑(commutatorSet G)) (Gr …
  -/
  obtain ⟨S, hS1, hS2⟩ := Group.rank_spec G
  /-
    case intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite ↑(commutatorSet G)
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    ⊢ LE.le (Subgroup.center G).index (HPow.hPow (Nat.card ↑(commutatorSet G)) (Gr …
  -/
  rw [← hS1, ← Fintype.card_coe, ← Nat.card_eq_fintype_card, ← Finset.coe_sort_coe, ← Nat.card_fun]
  /-
    case intro.intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite ↑(commutatorSet G)
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    ⊢ LE.le (Subgroup.center G).index (Nat.card (↑↑S → ↑(commutatorSet G)))
  -/
  exact Finite.card_le_of_embedding (quotientCenterEmbedding hS2)
  /-
    🎉 no goals
  -/


theorem index_stabilizer :
    (stabilizer G x).index = (orbit G x).ncard :=
  (Nat.card_congr (MulAction.orbitEquivQuotientStabilizer G x)).symm.trans
    (Set.Nat.card_coe_set_eq (orbit G x))


theorem index_stabilizer_of_transitive [IsPretransitive G X] :
    (stabilizer G x).index = Nat.card X := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G X
    x : X
    inst✝ : MulAction.IsPretransitive G X
    ⊢ Eq (MulAction.stabilizer G x).index (Nat.card X)
  -/
  rw [index_stabilizer, orbit_eq_univ, Set.ncard_univ]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma card_fiber_eq_of_mem_range (f : F) {x y : M} (hx : x ∈ Set.range f) (hy : y ∈ Set.range f) :
    #{g | f g = x} = #{g | f g = y} := by
  /-
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y : M
    hx : Membership.mem (Set.range ⇑f) x
    hy : Membership.mem (Set.range ⇑f) y
    ⊢ Eq (Finset.filter (fun g => Eq (f g) x) Finset.univ).card (Finset.filter (fu …
  -/
  rcases hx with ⟨x, rfl⟩
  /-
    case intro
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    y : M
    hy : Membership.mem (Set.range ⇑f) y
    x : G
    ⊢ Eq (Finset.filter (fun g => Eq (f g) (f x)) Finset.univ).card (Finset.filter …
  -/
  rcases hy with ⟨y, rfl⟩
  /-
    case intro.intro
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y : G
    ⊢ Eq (Finset.filter (fun g => Eq (f g) (f x)) Finset.univ).card (Finset.filter …
  -/
  rcases mul_left_surjective x y with ⟨y, rfl⟩
  conv_lhs =>
    rw [← map_univ_equiv (Equiv.mulRight y⁻¹), filter_map, card_map]
  /-
    case intro.intro.intro
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y : G
    ⊢ Eq (Finset.filter (Function.comp (fun g => Eq (f g) (f x)) ⇑(Equiv.toEmbeddi …
  -/
  congr 2 with g
  /-
    case intro.intro.intro.e_s.e_p.h.a
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y g : G
    ⊢ Iff (Function.comp (fun g => Eq (f g) (f x)) (⇑(Equiv.toEmbedding (Equiv.mul …
  -/
  simp only [Function.comp, Equiv.toEmbedding_apply, Equiv.coe_mulRight, map_mul]
  /-
    case intro.intro.intro.e_s.e_p.h.a
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y g : G
    ⊢ Iff (Eq (HMul.hMul (f g) (f (Inv.inv y))) (f x)) (Eq (f g) (HMul.hMul (f x)  …
  -/
  let f' := MonoidHomClass.toMonoidHom f
  /-
    case intro.intro.intro.e_s.e_p.h.a
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y g : G
    f' : MonoidHom G M := ↑f
    ⊢ Iff (Eq (HMul.hMul (f g) (f (Inv.inv y))) (f x)) (Eq (f g) (HMul.hMul (f x)  …
  -/
  show f' g * f' y⁻¹ = f' x ↔ f' g = f' x * f' y
  /-
    case intro.intro.intro.e_s.e_p.h.a
    G : Type u_1
    M : Type u_2
    F : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Fintype G
    inst✝³ : Monoid M
    inst✝² : DecidableEq M
    inst✝¹ : FunLike F G M
    inst✝ : MonoidHomClass F G M
    f : F
    x y g : G
    f' : MonoidHom G M := ↑f
    ⊢ Iff (Eq (HMul.hMul (f' g) (f' (Inv.inv y))) (f' x)) (Eq (f' g) (HMul.hMul (f …
  -/
  rw [← f'.coe_toHomUnits y⁻¹, map_inv, Units.mul_inv_eq_iff_eq_mul, f'.coe_toHomUnits]
  /-
    🎉 no goals
  -/



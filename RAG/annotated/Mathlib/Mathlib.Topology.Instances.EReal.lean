instance : TopologicalSpace EReal := Preorder.topology EReal

instance : OrderTopology EReal := ⟨rfl⟩

instance : T5Space EReal := inferInstance

instance : T2Space EReal := inferInstance


lemma denseRange_ratCast : DenseRange (fun r : ℚ ↦ ((r : ℝ) : EReal)) :=
  dense_of_exists_between fun _ _ h => exists_range_iff.2 <| exists_rat_btwn_of_lt h


instance : SecondCountableTopology EReal :=
  have : SeparableSpace EReal := ⟨⟨_, countable_range _, denseRange_ratCast⟩⟩
  .of_separableSpace_orderTopology _


theorem isEmbedding_coe : IsEmbedding ((↑) : ℝ → EReal) :=
                                                   /-
                                                     ⊢ (Set.range Real.toEReal).OrdConnected
                                                   -/
  coe_strictMono.isEmbedding_of_ordConnected <| by rw [range_coe_eq_Ioo]; exact ordConnected_Ioo
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[deprecated (since := "2024-10-26")]
alias embedding_coe := isEmbedding_coe


theorem isOpenEmbedding_coe : IsOpenEmbedding ((↑) : ℝ → EReal) :=
                       /-
                         ⊢ IsOpen (Set.range Real.toEReal)
                       -/
  ⟨isEmbedding_coe, by simp only [range_coe_eq_Ioo, isOpen_Ioo]⟩
                       /-
                         🎉 no goals
                       -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_coe := isOpenEmbedding_coe


@[norm_cast]
theorem tendsto_coe {α : Type*} {f : Filter α} {m : α → ℝ} {a : ℝ} :
    Tendsto (fun a => (m a : EReal)) f (𝓝 ↑a) ↔ Tendsto m f (𝓝 a) :=
  isEmbedding_coe.tendsto_nhds_iff.symm


theorem _root_.continuous_coe_real_ereal : Continuous ((↑) : ℝ → EReal) :=
  isEmbedding_coe.continuous


theorem continuous_coe_iff {f : α → ℝ} : (Continuous fun a => (f a : EReal)) ↔ Continuous f :=
  isEmbedding_coe.continuous_iff.symm


theorem nhds_coe {r : ℝ} : 𝓝 (r : EReal) = (𝓝 r).map (↑) :=
  (isOpenEmbedding_coe.map_nhds_eq r).symm


theorem nhds_coe_coe {r p : ℝ} :
    𝓝 ((r : EReal), (p : EReal)) = (𝓝 (r, p)).map fun p : ℝ × ℝ => (↑p.1, ↑p.2) :=
  ((isOpenEmbedding_coe.prodMap isOpenEmbedding_coe).map_nhds_eq (r, p)).symm


theorem tendsto_toReal {a : EReal} (ha : a ≠ ⊤) (h'a : a ≠ ⊥) :
    Tendsto EReal.toReal (𝓝 a) (𝓝 a.toReal) := by
  /-
    a : EReal
    ha : Ne a Top.top
    h'a : Ne a Bot.bot
    ⊢ Filter.Tendsto EReal.toReal (nhds a) (nhds a.toReal)
  -/
  lift a to ℝ using ⟨ha, h'a⟩
  /-
    case intro
    a : Real
    ha : Ne (↑a) Top.top
    h'a : Ne (↑a) Bot.bot
    ⊢ Filter.Tendsto EReal.toReal (nhds ↑a) (nhds (↑a).toReal)
  -/
  rw [nhds_coe, tendsto_map'_iff]
  /-
    case intro
    a : Real
    ha : Ne (↑a) Top.top
    h'a : Ne (↑a) Bot.bot
    ⊢ Filter.Tendsto (Function.comp EReal.toReal Real.toEReal) (nhds a) (nhds (↑a) …
  -/
  exact tendsto_id
  /-
    🎉 no goals
  -/


theorem continuousOn_toReal : ContinuousOn EReal.toReal ({⊥, ⊤}ᶜ : Set EReal) := fun _a ha =>
  ContinuousAt.continuousWithinAt (tendsto_toReal (mt Or.inr ha) (mt Or.inl ha))


/-- The set of finite `EReal` numbers is homeomorphic to `ℝ`. -/
def neBotTopHomeomorphReal : ({⊥, ⊤}ᶜ : Set EReal) ≃ₜ ℝ where
  toEquiv := neTopBotEquivReal
  continuous_toFun := continuousOn_iff_continuous_restrict.1 continuousOn_toReal
  continuous_invFun := continuous_coe_real_ereal.subtype_mk _


theorem isEmbedding_coe_ennreal : IsEmbedding ((↑) : ℝ≥0∞ → EReal) :=
  coe_ennreal_strictMono.isEmbedding_of_ordConnected <| by
    /-
      ⊢ (Set.range ENNReal.toEReal).OrdConnected
    -/
    rw [range_coe_ennreal]; exact ordConnected_Ici
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-10-26")]
alias embedding_coe_ennreal := isEmbedding_coe_ennreal


theorem isClosedEmbedding_coe_ennreal : IsClosedEmbedding ((↑) : ℝ≥0∞ → EReal) :=
                               /-
                                 ⊢ IsClosed (Set.range ENNReal.toEReal)
                               -/
  ⟨isEmbedding_coe_ennreal, by rw [range_coe_ennreal]; exact isClosed_Ici⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_coe_ennreal := isClosedEmbedding_coe_ennreal


@[norm_cast]
theorem tendsto_coe_ennreal {α : Type*} {f : Filter α} {m : α → ℝ≥0∞} {a : ℝ≥0∞} :
    Tendsto (fun a => (m a : EReal)) f (𝓝 ↑a) ↔ Tendsto m f (𝓝 a) :=
  isEmbedding_coe_ennreal.tendsto_nhds_iff.symm


theorem _root_.continuous_coe_ennreal_ereal : Continuous ((↑) : ℝ≥0∞ → EReal) :=
  isEmbedding_coe_ennreal.continuous


theorem continuous_coe_ennreal_iff {f : α → ℝ≥0∞} :
    (Continuous fun a => (f a : EReal)) ↔ Continuous f :=
  isEmbedding_coe_ennreal.continuous_iff.symm


theorem nhds_top : 𝓝 (⊤ : EReal) = ⨅ (a) (_ : a ≠ ⊤), 𝓟 (Ioi a) :=
                             /-
                               ⊢ Eq (iInf fun l => iInf fun x => Filter.principal (Set.Ioi l)) (iInf fun a => …
                             -/
  nhds_top_order.trans <| by simp only [lt_top_iff_ne_top]
                             /-
                               🎉 no goals
                             -/


nonrec theorem nhds_top_basis : (𝓝 (⊤ : EReal)).HasBasis (fun _ : ℝ ↦ True) (Ioi ·) := by
  /-
    ⊢ (nhds Top.top).HasBasis (fun x => True) fun x => Set.Ioi ↑x
  -/
  refine nhds_top_basis.to_hasBasis (fun x hx => ?_) fun _ _ ↦ ⟨_, coe_lt_top _, Subset.rfl⟩
  /-
    x : EReal
    hx : LT.lt x Top.top
    ⊢ Exists fun i' => And True (HasSubset.Subset (Set.Ioi ↑i') (Set.Ioi x))
  -/
  rcases exists_rat_btwn_of_lt hx with ⟨y, hxy, -⟩
  /-
    case intro.intro
    x : EReal
    hx : LT.lt x Top.top
    y : Rat
    hxy : LT.lt x ↑↑y
    ⊢ Exists fun i' => And True (HasSubset.Subset (Set.Ioi ↑i') (Set.Ioi x))
  -/
  exact ⟨_, trivial, Ioi_subset_Ioi hxy.le⟩
  /-
    🎉 no goals
  -/


theorem nhds_top' : 𝓝 (⊤ : EReal) = ⨅ a : ℝ, 𝓟 (Ioi ↑a) := nhds_top_basis.eq_iInf


theorem mem_nhds_top_iff {s : Set EReal} : s ∈ 𝓝 (⊤ : EReal) ↔ ∃ y : ℝ, Ioi (y : EReal) ⊆ s :=
                                     /-
                                       s : Set EReal
                                       ⊢ Iff (Exists fun i => And True (HasSubset.Subset (Set.Ioi ↑i) s)) (Exists fun …
                                     -/
  nhds_top_basis.mem_iff.trans <| by simp only [true_and]
                                     /-
                                       🎉 no goals
                                     -/


theorem tendsto_nhds_top_iff_real {α : Type*} {m : α → EReal} {f : Filter α} :
    Tendsto m f (𝓝 ⊤) ↔ ∀ x : ℝ, ∀ᶠ a in f, ↑x < m a :=
                                               /-
                                                 α : Type u_2
                                                 m : α → EReal
                                                 f : Filter α
                                                 ⊢ Iff (∀ (i : Real), True → Filter.Eventually (fun x => Membership.mem (Set.Io …
                                               -/
  nhds_top_basis.tendsto_right_iff.trans <| by simp only [true_implies, mem_Ioi]
                                               /-
                                                 🎉 no goals
                                               -/


theorem nhds_bot : 𝓝 (⊥ : EReal) = ⨅ (a) (_ : a ≠ ⊥), 𝓟 (Iio a) :=
                             /-
                               ⊢ Eq (iInf fun l => iInf fun x => Filter.principal (Set.Iio l)) (iInf fun a => …
                             -/
  nhds_bot_order.trans <| by simp only [bot_lt_iff_ne_bot]
                             /-
                               🎉 no goals
                             -/


theorem nhds_bot_basis : (𝓝 (⊥ : EReal)).HasBasis (fun _ : ℝ ↦ True) (Iio ·) := by
  /-
    ⊢ (nhds Bot.bot).HasBasis (fun x => True) fun x => Set.Iio ↑x
  -/
  refine _root_.nhds_bot_basis.to_hasBasis (fun x hx => ?_) fun _ _ ↦ ⟨_, bot_lt_coe _, Subset.rfl⟩
  /-
    x : EReal
    hx : LT.lt Bot.bot x
    ⊢ Exists fun i' => And True (HasSubset.Subset (Set.Iio ↑i') (Set.Iio x))
  -/
  rcases exists_rat_btwn_of_lt hx with ⟨y, -, hxy⟩
  /-
    case intro.intro
    x : EReal
    hx : LT.lt Bot.bot x
    y : Rat
    hxy : LT.lt (↑↑y) x
    ⊢ Exists fun i' => And True (HasSubset.Subset (Set.Iio ↑i') (Set.Iio x))
  -/
  exact ⟨_, trivial, Iio_subset_Iio hxy.le⟩
  /-
    🎉 no goals
  -/


theorem nhds_bot' : 𝓝 (⊥ : EReal) = ⨅ a : ℝ, 𝓟 (Iio ↑a) :=
  nhds_bot_basis.eq_iInf


theorem mem_nhds_bot_iff {s : Set EReal} : s ∈ 𝓝 (⊥ : EReal) ↔ ∃ y : ℝ, Iio (y : EReal) ⊆ s :=
                                     /-
                                       s : Set EReal
                                       ⊢ Iff (Exists fun i => And True (HasSubset.Subset (Set.Iio ↑i) s)) (Exists fun …
                                     -/
  nhds_bot_basis.mem_iff.trans <| by simp only [true_and]
                                     /-
                                       🎉 no goals
                                     -/


theorem tendsto_nhds_bot_iff_real {α : Type*} {m : α → EReal} {f : Filter α} :
    Tendsto m f (𝓝 ⊥) ↔ ∀ x : ℝ, ∀ᶠ a in f, m a < x :=
                                               /-
                                                 α : Type u_2
                                                 m : α → EReal
                                                 f : Filter α
                                                 ⊢ Iff (∀ (i : Real), True → Filter.Eventually (fun x => Membership.mem (Set.Ii …
                                               -/
  nhds_bot_basis.tendsto_right_iff.trans <| by simp only [true_implies, mem_Iio]
                                               /-
                                                 🎉 no goals
                                               -/


lemma nhdsWithin_top : 𝓝[≠] (⊤ : EReal) = (atTop).map Real.toEReal := by
  /-
    ⊢ Eq (nhdsWithin Top.top (HasCompl.compl (Singleton.singleton Top.top))) (Filt …
  -/
  apply (nhdsWithin_hasBasis nhds_top_basis_Ici _).ext (atTop_basis.map Real.toEReal)
    /-
      case h
      ⊢ ∀ (i : EReal), LT.lt i Top.top → Exists fun i' => And True (HasSubset.Subset …
    -/
  · simp only [EReal.image_coe_Ici, true_and]
    /-
      case h
      ⊢ ∀ (i : EReal), LT.lt i Top.top → Exists fun i' => HasSubset.Subset (Set.Ico  …
    -/
    intro x hx
    /-
      case h
      x : EReal
      hx : LT.lt x Top.top
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ico (↑i') Top.top) (Inter.inter (Set. …
    -/
    by_cases hx_bot : x = ⊥
      /-
        case pos
        x : EReal
        hx : LT.lt x Top.top
        hx_bot : Eq x Bot.bot
        ⊢ Exists fun i' => HasSubset.Subset (Set.Ico (↑i') Top.top) (Inter.inter (Set. …
      -/
    · simp [hx_bot]
      /-
        🎉 no goals
      -/
    /-
      case neg
      x : EReal
      hx : LT.lt x Top.top
      hx_bot : Not (Eq x Bot.bot)
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ico (↑i') Top.top) (Inter.inter (Set. …
    -/
    lift x to ℝ using ⟨hx.ne_top, hx_bot⟩
    /-
      case neg.intro
      x : Real
      hx : LT.lt (↑x) Top.top
      hx_bot : Not (Eq (↑x) Bot.bot)
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ico (↑i') Top.top) (Inter.inter (Set. …
    -/
    refine ⟨x, fun x ⟨h1, h2⟩ ↦ ?_⟩
    /-
      case neg.intro
      x✝¹ : Real
      hx : LT.lt (↑x✝¹) Top.top
      hx_bot : Not (Eq (↑x✝¹) Bot.bot)
      x : EReal
      x✝ : Membership.mem (Set.Ico (↑x✝¹) Top.top) x
      h1 : LE.le (↑x✝¹) x
      h2 : LT.lt x Top.top
      ⊢ Membership.mem (Inter.inter (Set.Ici ↑x✝¹) (HasCompl.compl (Singleton.single …
    -/
    simp [h1, h2.ne_top]
    /-
      🎉 no goals
    -/
    /-
      case h'
      ⊢ ∀ (i' : Real), True → Exists fun i => And (LT.lt i Top.top) (HasSubset.Subse …
    -/
  · simp only [EReal.image_coe_Ici, true_implies]
    /-
      case h'
      ⊢ ∀ (i' : Real), Exists fun i => And (LT.lt i Top.top) (HasSubset.Subset (Inte …
    -/
    refine fun x ↦ ⟨x, ⟨EReal.coe_lt_top x, fun x ⟨(h1 : _ ≤ x), h2⟩ ↦ ?_⟩⟩
    /-
      case h'
      x✝¹ : Real
      x : EReal
      x✝ : Membership.mem (Inter.inter (Set.Ici ↑x✝¹) (HasCompl.compl (Singleton.sin …
      h1 : LE.le (↑x✝¹) x
      h2 : Membership.mem (HasCompl.compl (Singleton.singleton Top.top)) x
      ⊢ Membership.mem (Set.Ico (↑x✝¹) Top.top) x
    -/
    simp [h1, Ne.lt_top' fun a ↦ h2 a.symm]
    /-
      🎉 no goals
    -/


lemma nhdsWithin_bot : 𝓝[≠] (⊥ : EReal) = (atBot).map Real.toEReal := by
  /-
    ⊢ Eq (nhdsWithin Bot.bot (HasCompl.compl (Singleton.singleton Bot.bot))) (Filt …
  -/
  apply (nhdsWithin_hasBasis nhds_bot_basis_Iic _).ext (atBot_basis.map Real.toEReal)
  · simp only [EReal.image_coe_Iic, Set.subset_compl_singleton_iff, Set.mem_Ioc, lt_self_iff_false,
      bot_le, and_true, not_false_eq_true, true_and]
    /-
      case h
      ⊢ ∀ (i : EReal), LT.lt Bot.bot i → Exists fun i' => HasSubset.Subset (Set.Ioc  …
    -/
    intro x hx
    /-
      case h
      x : EReal
      hx : LT.lt Bot.bot x
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ioc Bot.bot ↑i') (Inter.inter (Set.Ii …
    -/
    by_cases hx_top : x = ⊤
      /-
        case pos
        x : EReal
        hx : LT.lt Bot.bot x
        hx_top : Eq x Top.top
        ⊢ Exists fun i' => HasSubset.Subset (Set.Ioc Bot.bot ↑i') (Inter.inter (Set.Ii …
      -/
    · simp [hx_top]
      /-
        🎉 no goals
      -/
    /-
      case neg
      x : EReal
      hx : LT.lt Bot.bot x
      hx_top : Not (Eq x Top.top)
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ioc Bot.bot ↑i') (Inter.inter (Set.Ii …
    -/
    lift x to ℝ using ⟨hx_top, hx.ne_bot⟩
    /-
      case neg.intro
      x : Real
      hx : LT.lt Bot.bot ↑x
      hx_top : Not (Eq (↑x) Top.top)
      ⊢ Exists fun i' => HasSubset.Subset (Set.Ioc Bot.bot ↑i') (Inter.inter (Set.Ii …
    -/
    refine ⟨x, fun x ⟨h1, h2⟩ ↦ ?_⟩
    /-
      case neg.intro
      x✝¹ : Real
      hx : LT.lt Bot.bot ↑x✝¹
      hx_top : Not (Eq (↑x✝¹) Top.top)
      x : EReal
      x✝ : Membership.mem (Set.Ioc Bot.bot ↑x✝¹) x
      h1 : LT.lt Bot.bot x
      h2 : LE.le x ↑x✝¹
      ⊢ Membership.mem (Inter.inter (Set.Iic ↑x✝¹) (HasCompl.compl (Singleton.single …
    -/
    simp [h2, h1.ne_bot]
    /-
      🎉 no goals
    -/
    /-
      case h'
      ⊢ ∀ (i' : Real), True → Exists fun i => And (LT.lt Bot.bot i) (HasSubset.Subse …
    -/
  · simp only [EReal.image_coe_Iic, true_implies]
    /-
      case h'
      ⊢ ∀ (i' : Real), Exists fun i => And (LT.lt Bot.bot i) (HasSubset.Subset (Inte …
    -/
    refine fun x ↦ ⟨x, ⟨EReal.bot_lt_coe x, fun x ⟨(h1 : x ≤ _), h2⟩ ↦ ?_⟩⟩
    /-
      case h'
      x✝¹ : Real
      x : EReal
      x✝ : Membership.mem (Inter.inter (Set.Iic ↑x✝¹) (HasCompl.compl (Singleton.sin …
      h1 : LE.le x ↑x✝¹
      h2 : Membership.mem (HasCompl.compl (Singleton.singleton Bot.bot)) x
      ⊢ Membership.mem (Set.Ioc Bot.bot ↑x✝¹) x
    -/
    simp [h1, Ne.bot_lt' fun a ↦ h2 a.symm]
    /-
      🎉 no goals
    -/


lemma tendsto_toReal_atTop : Tendsto EReal.toReal (𝓝[≠] ⊤) atTop := by
  /-
    ⊢ Filter.Tendsto EReal.toReal (nhdsWithin Top.top (HasCompl.compl (Singleton.s …
  -/
  rw [nhdsWithin_top, tendsto_map'_iff]
  /-
    ⊢ Filter.Tendsto (Function.comp EReal.toReal Real.toEReal) Filter.atTop Filter …
  -/
  exact tendsto_id
  /-
    🎉 no goals
  -/


lemma tendsto_toReal_atBot : Tendsto EReal.toReal (𝓝[≠] ⊥) atBot := by
  /-
    ⊢ Filter.Tendsto EReal.toReal (nhdsWithin Bot.bot (HasCompl.compl (Singleton.s …
  -/
  rw [nhdsWithin_bot, tendsto_map'_iff]
  /-
    ⊢ Filter.Tendsto (Function.comp EReal.toReal Real.toEReal) Filter.atBot Filter …
  -/
  exact tendsto_id
  /-
    🎉 no goals
  -/


lemma add_iInf_le_iInf_add : (⨅ x, u x) + ⨅ x, v x ≤ ⨅ x, (u + v) x :=
  le_iInf fun i ↦ add_le_add (iInf_le u i) (iInf_le v i)


lemma iSup_add_le_add_iSup : ⨆ x, (u + v) x ≤ (⨆ x, u x) + ⨆ x, v x :=
  iSup_le fun i ↦ add_le_add (le_iSup u i) (le_iSup v i)


lemma liminf_neg : liminf (- v) f = - limsup v f :=
  /-
    α : Type u_3
    f : Filter α
    v : α → EReal
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  EReal.negOrderIso.limsup_apply.symm
  /-
    🎉 no goals
  -/


lemma limsup_neg : limsup (- v) f = - liminf v f :=
  /-
    α : Type u_3
    f : Filter α
    v : α → EReal
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  EReal.negOrderIso.liminf_apply.symm
  /-
    🎉 no goals
  -/


lemma le_liminf_add : (liminf u f) + (liminf v f) ≤ liminf (u + v) f := by
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    ⊢ LE.le (HAdd.hAdd (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HA …
  -/
  refine add_le_of_forall_lt fun a a_u b b_v ↦ (le_liminf_iff).2 fun c c_ab ↦ ?_
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    a : EReal
    a_u : LT.lt a (Filter.liminf u f)
    b : EReal
    b_v : LT.lt b (Filter.liminf v f)
    c : EReal
    c_ab : LT.lt c (HAdd.hAdd a b)
    ⊢ Filter.Eventually (fun a => LT.lt c (HAdd.hAdd u v a)) f
  -/
  filter_upwards [eventually_lt_of_lt_liminf a_u, eventually_lt_of_lt_liminf b_v] with x a_x b_x
  /-
    case h
    α : Type u_3
    f : Filter α
    u v : α → EReal
    a : EReal
    a_u : LT.lt a (Filter.liminf u f)
    b : EReal
    b_v : LT.lt b (Filter.liminf v f)
    c : EReal
    c_ab : LT.lt c (HAdd.hAdd a b)
    x : α
    a_x : LT.lt a (u x)
    b_x : LT.lt b (v x)
    ⊢ LT.lt c (HAdd.hAdd u v x)
  -/
  exact c_ab.trans (add_lt_add a_x b_x)
  /-
    🎉 no goals
  -/


lemma limsup_add_le (h : limsup u f ≠ ⊥ ∨ limsup v f ≠ ⊤) (h' : limsup u f ≠ ⊤ ∨ limsup v f ≠ ⊥) :
    limsup (u + v) f ≤ (limsup u f) + (limsup v f) := by
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Or (Ne (Filter.limsup u f) Bot.bot) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) Bot.bot)
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  refine le_add_of_forall_gt h h' fun a a_u b b_v ↦ (limsup_le_iff).2 fun c c_ab ↦ ?_
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Or (Ne (Filter.limsup u f) Bot.bot) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) Bot.bot)
    a : EReal
    a_u : GT.gt a (Filter.limsup u f)
    b : EReal
    b_v : GT.gt b (Filter.limsup v f)
    c : EReal
    c_ab : GT.gt c (HAdd.hAdd a b)
    ⊢ Filter.Eventually (fun a => LT.lt (HAdd.hAdd u v a) c) f
  -/
  filter_upwards [eventually_lt_of_limsup_lt a_u, eventually_lt_of_limsup_lt b_v] with x a_x b_x
  /-
    case h
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Or (Ne (Filter.limsup u f) Bot.bot) (Ne (Filter.limsup v f) Top.top)
    h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) Bot.bot)
    a : EReal
    a_u : GT.gt a (Filter.limsup u f)
    b : EReal
    b_v : GT.gt b (Filter.limsup v f)
    c : EReal
    c_ab : GT.gt c (HAdd.hAdd a b)
    x : α
    a_x : LT.lt (u x) a
    b_x : LT.lt (v x) b
    ⊢ LT.lt (HAdd.hAdd u v x) c
  -/
  exact (add_lt_add a_x b_x).trans c_ab
  /-
    🎉 no goals
  -/


lemma le_limsup_add : (limsup u f) + (liminf v f) ≤ limsup (u + v) f :=
                                         /-
                                           α : Type u_3
                                           f : Filter α
                                           u v : α → EReal
                                           x✝¹ : EReal
                                           a_u : LT.lt x✝¹ (Filter.limsup u f)
                                           x✝ : EReal
                                           b_v : LT.lt x✝ (Filter.liminf v f)
                                           ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  add_le_of_forall_lt fun _ a_u _ b_v ↦ (le_limsup_iff).2 fun _ c_ab ↦
                                         /-
                                           🎉 no goals
                                         -/
       /-
         α : Type u_3
         f : Filter α
         u v : α → EReal
         x✝² : EReal
         a_u : LT.lt x✝² (Filter.limsup u f)
         x✝¹ : EReal
         b_v : LT.lt x✝¹ (Filter.liminf v f)
         x✝ : EReal
         c_ab : LT.lt x✝ (HAdd.hAdd x✝² x✝¹)
         ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u
       -/
       /-
         🎉 no goals
       -/
    (((frequently_lt_of_lt_limsup) a_u).and_eventually ((eventually_lt_of_lt_liminf) b_v)).mono
                                                         /-
                                                           🎉 no goals
                                                         -/
    fun _ ab_x ↦ c_ab.trans (add_lt_add ab_x.1 ab_x.2)


lemma liminf_add_le (h : limsup u f ≠ ⊥ ∨ liminf v f ≠ ⊤) (h' : limsup u f ≠ ⊤ ∨ liminf v f ≠ ⊥) :
    liminf (u + v) f ≤ (limsup u f) + (liminf v f) :=
                                              /-
                                                α : Type u_3
                                                f : Filter α
                                                u v : α → EReal
                                                h : Or (Ne (Filter.limsup u f) Bot.bot) (Ne (Filter.liminf v f) Top.top)
                                                h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.liminf v f) Bot.bot)
                                                x✝¹ : EReal
                                                a_u : GT.gt x✝¹ (Filter.limsup u f)
                                                x✝ : EReal
                                                b_v : GT.gt x✝ (Filter.liminf v f)
                                                ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  le_add_of_forall_gt h h' fun _ a_u _ b_v ↦ (liminf_le_iff).2 fun _ c_ab ↦
                                              /-
                                                🎉 no goals
                                              -/
       /-
         α : Type u_3
         f : Filter α
         u v : α → EReal
         h : Or (Ne (Filter.limsup u f) Bot.bot) (Ne (Filter.liminf v f) Top.top)
         h' : Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.liminf v f) Bot.bot)
         x✝² : EReal
         a_u : GT.gt x✝² (Filter.limsup u f)
         x✝¹ : EReal
         b_v : GT.gt x✝¹ (Filter.liminf v f)
         x✝ : EReal
         c_ab : GT.gt x✝ (HAdd.hAdd x✝² x✝¹)
         ⊢ Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
       -/
       /-
         🎉 no goals
       -/
    (((frequently_lt_of_liminf_lt) b_v).and_eventually ((eventually_lt_of_limsup_lt) a_u)).mono
                                                         /-
                                                           🎉 no goals
                                                         -/
    fun _ ab_x ↦ (add_lt_add ab_x.2 ab_x.1).trans c_ab


@[deprecated (since := "2024-11-11")] alias add_liminf_le_liminf_add := le_liminf_add

@[deprecated (since := "2024-11-11")] alias limsup_add_le_add_limsup := limsup_add_le

@[deprecated (since := "2024-11-11")] alias limsup_add_liminf_le_limsup_add := le_limsup_add

@[deprecated (since := "2024-11-11")] alias liminf_add_le_limsup_add_liminf := liminf_add_le


lemma limsup_add_bot_of_ne_top (h : limsup u f = ⊥) (h' : limsup v f ≠ ⊤) :
    limsup (u + v) f = ⊥ := by
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Eq (Filter.limsup u f) Bot.bot
    h' : Ne (Filter.limsup v f) Top.top
    ⊢ Eq (Filter.limsup (HAdd.hAdd u v) f) Bot.bot
  -/
  apply le_bot_iff.1 ((limsup_add_le (.inr h') _).trans _)
    /-
      α : Type u_3
      f : Filter α
      u v : α → EReal
      h : Eq (Filter.limsup u f) Bot.bot
      h' : Ne (Filter.limsup v f) Top.top
      ⊢ Or (Ne (Filter.limsup u f) Top.top) (Ne (Filter.limsup v f) Bot.bot)
    -/
  · rw [h]; exact .inl bot_ne_top
            /-
              🎉 no goals
            -/
    /-
      α : Type u_3
      f : Filter α
      u v : α → EReal
      h : Eq (Filter.limsup u f) Bot.bot
      h' : Ne (Filter.limsup v f) Top.top
      ⊢ LE.le (HAdd.hAdd (Filter.limsup u f) (Filter.limsup v f)) Bot.bot
    -/
  · rw [h, bot_add]
    /-
      🎉 no goals
    -/


lemma limsup_add_le_of_le (ha : limsup u f < a) (hb : limsup v f ≤ b) :
    limsup (u + v) f ≤ a + b := by
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    a b : EReal
    ha : LT.lt (Filter.limsup u f) a
    hb : LE.le (Filter.limsup v f) b
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd a b)
  -/
  rcases eq_top_or_lt_top b with rfl | h
    /-
      case inl
      α : Type u_3
      f : Filter α
      u v : α → EReal
      a : EReal
      ha : LT.lt (Filter.limsup u f) a
      hb : LE.le (Filter.limsup v f) Top.top
      ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd a Top.top)
    -/
  · rw [add_top_of_ne_bot ha.ne_bot]; exact le_top
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case inr
      α : Type u_3
      f : Filter α
      u v : α → EReal
      a b : EReal
      ha : LT.lt (Filter.limsup u f) a
      hb : LE.le (Filter.limsup v f) b
      h : LT.lt b Top.top
      ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd a b)
    -/
  · exact (limsup_add_le (.inr (hb.trans_lt h).ne) (.inl ha.ne_top)).trans (add_le_add ha.le hb)
    /-
      🎉 no goals
    -/


lemma liminf_add_gt_of_gt (ha : a < liminf u f) (hb : b < liminf v f) :
    a + b < liminf (u + v) f :=
  (add_lt_add ha hb).trans_le le_liminf_add


lemma liminf_add_top_of_ne_bot (h : liminf u f = ⊤) (h' : liminf v f ≠ ⊥) :
    liminf (u + v) f = ⊤ := by
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Eq (Filter.liminf u f) Top.top
    h' : Ne (Filter.liminf v f) Bot.bot
    ⊢ Eq (Filter.liminf (HAdd.hAdd u v) f) Top.top
  -/
  apply top_le_iff.1 (le_trans _ le_liminf_add)
  /-
    α : Type u_3
    f : Filter α
    u v : α → EReal
    h : Eq (Filter.liminf u f) Top.top
    h' : Ne (Filter.liminf v f) Bot.bot
    ⊢ LE.le Top.top (HAdd.hAdd (Filter.liminf u f) (Filter.liminf v f))
  -/
  rw [h, top_add_of_ne_bot h']
  /-
    🎉 no goals
  -/


theorem continuousAt_add_coe_coe (a b : ℝ) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (a, b) := by
  simp only [ContinuousAt, nhds_coe_coe, ← coe_add, tendsto_map'_iff, Function.comp_def,
    tendsto_coe, tendsto_add]


theorem continuousAt_add_top_coe (a : ℝ) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (⊤, a) := by
  /-
    a : Real
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Top.top, snd := ↑a }
  -/
  simp only [ContinuousAt, tendsto_nhds_top_iff_real, top_add_coe]
  refine fun r ↦ ((lt_mem_nhds (coe_lt_top (r - (a - 1)))).prod_nhds
    (lt_mem_nhds <| EReal.coe_lt_coe_iff.2 <| sub_one_lt _)).mono fun _ h ↦ ?_
  /-
    a r : Real
    x✝ : Prod EReal EReal
    h : And (LT.lt (↑(HSub.hSub r (HSub.hSub a 1))) x✝.1) (LT.lt (↑(HSub.hSub a 1) …
    ⊢ LT.lt (↑r) (HAdd.hAdd x✝.1 x✝.2)
  -/
  simpa only [← coe_add, _root_.sub_add_cancel] using add_lt_add h.1 h.2
  /-
    🎉 no goals
  -/


theorem continuousAt_add_coe_top (a : ℝ) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (a, ⊤) := by
  simpa only [add_comm, Function.comp_def, ContinuousAt, Prod.swap]
    using Tendsto.comp (continuousAt_add_top_coe a) (continuous_swap.tendsto ((a : EReal), ⊤))


theorem continuousAt_add_top_top : ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (⊤, ⊤) := by
  /-
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Top.top, snd := Top.top }
  -/
  simp only [ContinuousAt, tendsto_nhds_top_iff_real, top_add_top]
  refine fun r ↦ ((lt_mem_nhds (coe_lt_top 0)).prod_nhds
    (lt_mem_nhds <| coe_lt_top r)).mono fun _ h ↦ ?_
  /-
    r : Real
    x✝ : Prod EReal EReal
    h : And (LT.lt (↑0) x✝.1) (LT.lt (↑r) x✝.2)
    ⊢ LT.lt (↑r) (HAdd.hAdd x✝.1 x✝.2)
  -/
  simpa only [coe_zero, zero_add] using add_lt_add h.1 h.2
  /-
    🎉 no goals
  -/


theorem continuousAt_add_bot_coe (a : ℝ) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (⊥, a) := by
  /-
    a : Real
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Bot.bot, snd := ↑a }
  -/
  simp only [ContinuousAt, tendsto_nhds_bot_iff_real, bot_add]
  refine fun r ↦ ((gt_mem_nhds (bot_lt_coe (r - (a + 1)))).prod_nhds
    (gt_mem_nhds <| EReal.coe_lt_coe_iff.2 <| lt_add_one _)).mono fun _ h ↦ ?_
  /-
    a r : Real
    x✝ : Prod EReal EReal
    h : And (LT.lt x✝.1 ↑(HSub.hSub r (HAdd.hAdd a 1))) (LT.lt x✝.2 ↑(HAdd.hAdd a  …
    ⊢ LT.lt (HAdd.hAdd x✝.1 x✝.2) ↑r
  -/
  simpa only [← coe_add, _root_.sub_add_cancel] using add_lt_add h.1 h.2
  /-
    🎉 no goals
  -/


theorem continuousAt_add_coe_bot (a : ℝ) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (a, ⊥) := by
  simpa only [add_comm, Function.comp_def, ContinuousAt, Prod.swap]
    using Tendsto.comp (continuousAt_add_bot_coe a) (continuous_swap.tendsto ((a : EReal), ⊥))


theorem continuousAt_add_bot_bot : ContinuousAt (fun p : EReal × EReal => p.1 + p.2) (⊥, ⊥) := by
  /-
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Bot.bot, snd := Bot.bot }
  -/
  simp only [ContinuousAt, tendsto_nhds_bot_iff_real, bot_add]
  refine fun r ↦ ((gt_mem_nhds (bot_lt_coe 0)).prod_nhds
    (gt_mem_nhds <| bot_lt_coe r)).mono fun _ h ↦ ?_
  /-
    r : Real
    x✝ : Prod EReal EReal
    h : And (LT.lt x✝.1 ↑0) (LT.lt x✝.2 ↑r)
    ⊢ LT.lt (HAdd.hAdd x✝.1 x✝.2) ↑r
  -/
  simpa only [coe_zero, zero_add] using add_lt_add h.1 h.2
  /-
    🎉 no goals
  -/


/-- The addition on `EReal` is continuous except where it doesn't make sense (i.e., at `(⊥, ⊤)`
and at `(⊤, ⊥)`). -/
theorem continuousAt_add {p : EReal × EReal} (h : p.1 ≠ ⊤ ∨ p.2 ≠ ⊥) (h' : p.1 ≠ ⊥ ∨ p.2 ≠ ⊤) :
    ContinuousAt (fun p : EReal × EReal => p.1 + p.2) p := by
  /-
    p : Prod EReal EReal
    h : Or (Ne p.1 Top.top) (Ne p.2 Bot.bot)
    h' : Or (Ne p.1 Bot.bot) (Ne p.2 Top.top)
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) p
  -/
  rcases p with ⟨x, y⟩
  /-
    case mk
    x y : EReal
    h : Or (Ne { fst := x, snd := y }.1 Top.top) (Ne { fst := x, snd := y }.2 Bot. …
    h' : Or (Ne { fst := x, snd := y }.1 Bot.bot) (Ne { fst := x, snd := y }.2 Top …
    ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := x, snd := y }
  -/
  induction x <;> induction y
    /-
      case mk.h_bot.h_bot
      h : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 Top.top) (Ne { fst := Bot.bot, …
      h' : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 Bot.bot) (Ne { fst := Bot.bot …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Bot.bot, snd := Bot.bot }
    -/
  · exact continuousAt_add_bot_bot
    /-
      🎉 no goals
    -/
    /-
      case mk.h_bot.h_real
      a✝ : Real
      h : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 Top.top) (Ne { fst := Bot.bot, snd …
      h' : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := Bot.bot, sn …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Bot.bot, snd := ↑a✝ }
    -/
  · exact continuousAt_add_bot_coe _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_bot.h_top
      h : Or (Ne { fst := Bot.bot, snd := Top.top }.1 Top.top) (Ne { fst := Bot.bot, …
      h' : Or (Ne { fst := Bot.bot, snd := Top.top }.1 Bot.bot) (Ne { fst := Bot.bot …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Bot.bot, snd := Top.top }
    -/
  · simp at h'
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_bot
      a✝ : Real
      h : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Top.top) (Ne { fst := ↑a✝, snd :=  …
      h' : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := ↑a✝, snd := Bot.bot }
    -/
  · exact continuousAt_add_coe_bot _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_real
      a✝¹ a✝ : Real
      h : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 Top.top) (Ne { fst := ↑a✝¹, snd := ↑a …
      h' : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := ↑a✝¹, snd := ↑ …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := ↑a✝¹, snd := ↑a✝ }
    -/
  · exact continuousAt_add_coe_coe _ _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_top
      a✝ : Real
      h : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Top.top) (Ne { fst := ↑a✝, snd :=  …
      h' : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := ↑a✝, snd := Top.top }
    -/
  · exact continuousAt_add_coe_top _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_bot
      h : Or (Ne { fst := Top.top, snd := Bot.bot }.1 Top.top) (Ne { fst := Top.top, …
      h' : Or (Ne { fst := Top.top, snd := Bot.bot }.1 Bot.bot) (Ne { fst := Top.top …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Top.top, snd := Bot.bot }
    -/
  · simp at h
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_real
      a✝ : Real
      h : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 Top.top) (Ne { fst := Top.top, snd …
      h' : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := Top.top, sn …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Top.top, snd := ↑a✝ }
    -/
  · exact continuousAt_add_top_coe _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_top
      h : Or (Ne { fst := Top.top, snd := Top.top }.1 Top.top) (Ne { fst := Top.top, …
      h' : Or (Ne { fst := Top.top, snd := Top.top }.1 Bot.bot) (Ne { fst := Top.top …
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := Top.top, snd := Top.top }
    -/
  · exact continuousAt_add_top_top
    /-
      🎉 no goals
    -/


instance : ContinuousNeg EReal := ⟨negOrderIso.continuous⟩


private lemma continuousAt_mul_swap {a b : EReal}
    (h : ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, b)) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (b, a) := by
  /-
    a b : EReal
    h : ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := a, snd := b }
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := b, snd := a }
  -/
  convert h.comp continuous_swap.continuousAt (x := (b, a))
  /-
    case h.e'_5.h
    a b : EReal
    h : ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := a, snd := b }
    x✝ : Prod EReal EReal
    ⊢ Eq (HMul.hMul x✝.1 x✝.2) (Function.comp (fun p => HMul.hMul p.1 p.2) Prod.sw …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


private lemma continuousAt_mul_symm1 {a b : EReal}
    (h : ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, b)) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (-a, b) := by
  have : (fun p : EReal × EReal ↦ p.1 * p.2) = (fun x : EReal ↦ -x)
      ∘ (fun p : EReal × EReal ↦ p.1 * p.2) ∘ (fun p : EReal × EReal ↦ (-p.1, p.2)) := by
    ext p
    simp
  /-
    a b : EReal
    h : ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := a, snd := b }
    this : Eq (fun p => HMul.hMul p.1 p.2) (Function.comp (fun x => Neg.neg x) (Fu …
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Neg.neg a, snd := b }
  -/
  rw [this]
  apply ContinuousAt.comp (Continuous.continuousAt continuous_neg)
    <| ContinuousAt.comp _ (ContinuousAt.prodMap (Continuous.continuousAt continuous_neg)
      (Continuous.continuousAt continuous_id))
  /-
    a b : EReal
    h : ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := a, snd := b }
    this : Eq (fun p => HMul.hMul p.1 p.2) (Function.comp (fun x => Neg.neg x) (Fu …
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) (Prod.map (fun a => Neg.neg a) id  …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


private lemma continuousAt_mul_symm2 {a b : EReal}
    (h : ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, b)) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, -b) :=
  continuousAt_mul_swap (continuousAt_mul_symm1 (continuousAt_mul_swap h))


private lemma continuousAt_mul_symm3 {a b : EReal}
    (h : ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, b)) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (-a, -b) :=
  continuousAt_mul_symm1 (continuousAt_mul_symm2 h)


private lemma continuousAt_mul_coe_coe (a b : ℝ) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (a, b) := by
  simp [ContinuousAt, EReal.nhds_coe_coe, ← EReal.coe_mul, Filter.tendsto_map'_iff,
    Function.comp_def, EReal.tendsto_coe, tendsto_mul]


private lemma continuousAt_mul_top_top :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (⊤, ⊤) := by
  /-
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := Top.top }
  -/
  simp only [ContinuousAt, EReal.top_mul_top, EReal.tendsto_nhds_top_iff_real]
  /-
    ⊢ ∀ (x : Real), Filter.Eventually (fun a => LT.lt (↑x) (HMul.hMul a.1 a.2)) (n …
  -/
  intro x
  /-
    x : Real
    ⊢ Filter.Eventually (fun a => LT.lt (↑x) (HMul.hMul a.1 a.2)) (nhds { fst := T …
  -/
  rw [_root_.eventually_nhds_iff]
  /-
    x : Real
    ⊢ Exists fun t => And (∀ (y : Prod EReal EReal), Membership.mem t y → LT.lt (↑ …
  -/
  use (Set.Ioi ((max x 0) : EReal)) ×ˢ (Set.Ioi 1)
  /-
    case h
    x : Real
    ⊢ And (∀ (y : Prod EReal EReal), Membership.mem (SProd.sprod (Set.Ioi (Max.max …
  -/
  split_ands
    /-
      case h.refine_1
      x : Real
      ⊢ ∀ (y : Prod EReal EReal), Membership.mem (SProd.sprod (Set.Ioi (Max.max (↑x) …
    -/
  · intros p p_in_prod
    /-
      case h.refine_1
      x : Real
      p : Prod EReal EReal
      p_in_prod : Membership.mem (SProd.sprod (Set.Ioi (Max.max (↑x) 0)) (Set.Ioi 1) …
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    simp only [Set.mem_prod, Set.mem_Ioi, max_lt_iff] at p_in_prod
    /-
      case h.refine_1
      x : Real
      p : Prod EReal EReal
      p_in_prod : And (And (LT.lt (↑x) p.1) (LT.lt 0 p.1)) (LT.lt 1 p.2)
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    rcases p_in_prod with ⟨⟨p1_gt_x, p1_pos⟩, p2_gt_1⟩
    /-
      case h.refine_1.intro.intro
      x : Real
      p : Prod EReal EReal
      p2_gt_1 : LT.lt 1 p.2
      p1_gt_x : LT.lt (↑x) p.1
      p1_pos : LT.lt 0 p.1
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    have := mul_le_mul_of_nonneg_left (le_of_lt p2_gt_1) (le_of_lt p1_pos)
    /-
      case h.refine_1.intro.intro
      x : Real
      p : Prod EReal EReal
      p2_gt_1 : LT.lt 1 p.2
      p1_gt_x : LT.lt (↑x) p.1
      p1_pos : LT.lt 0 p.1
      this : LE.le (HMul.hMul p.1 1) (HMul.hMul p.1 p.2)
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    rw [mul_one p.1] at this
    /-
      case h.refine_1.intro.intro
      x : Real
      p : Prod EReal EReal
      p2_gt_1 : LT.lt 1 p.2
      p1_gt_x : LT.lt (↑x) p.1
      p1_pos : LT.lt 0 p.1
      this : LE.le p.1 (HMul.hMul p.1 p.2)
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    exact lt_of_lt_of_le p1_gt_x this
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.refine_1
      x : Real
      ⊢ IsOpen (SProd.sprod (Set.Ioi (Max.max (↑x) 0)) (Set.Ioi 1))
    -/
  · exact IsOpen.prod isOpen_Ioi isOpen_Ioi
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.refine_2.refine_1
      x : Real
      ⊢ Membership.mem (Set.Ioi (Max.max (↑x) 0)) { fst := Top.top, snd := Top.top }.1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.refine_2.refine_2
      x : Real
      ⊢ Membership.mem (Set.Ioi 1) { fst := Top.top, snd := Top.top }.2
    -/
  · rw [Set.mem_Ioi, ← EReal.coe_one]; exact EReal.coe_lt_top 1
                                       /-
                                         🎉 no goals
                                       -/


private lemma continuousAt_mul_top_pos {a : ℝ} (h : 0 < a) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (⊤, a) := by
  /-
    a : Real
    h : LT.lt 0 a
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a }
  -/
  simp only [ContinuousAt, EReal.top_mul_coe_of_pos h, EReal.tendsto_nhds_top_iff_real]
  /-
    a : Real
    h : LT.lt 0 a
    ⊢ ∀ (x : Real), Filter.Eventually (fun a => LT.lt (↑x) (HMul.hMul a.1 a.2)) (n …
  -/
  intro x
  /-
    a : Real
    h : LT.lt 0 a
    x : Real
    ⊢ Filter.Eventually (fun a => LT.lt (↑x) (HMul.hMul a.1 a.2)) (nhds { fst := T …
  -/
  rw [_root_.eventually_nhds_iff]
  /-
    a : Real
    h : LT.lt 0 a
    x : Real
    ⊢ Exists fun t => And (∀ (y : Prod EReal EReal), Membership.mem t y → LT.lt (↑ …
  -/
  use (Set.Ioi ((2*(max (x+1) 0)/a : ℝ) : EReal)) ×ˢ (Set.Ioi ((a/2 : ℝ) : EReal))
  /-
    case h
    a : Real
    h : LT.lt 0 a
    x : Real
    ⊢ And (∀ (y : Prod EReal EReal), Membership.mem (SProd.sprod (Set.Ioi ↑(HDiv.h …
  -/
  split_ands
    /-
      case h.refine_1
      a : Real
      h : LT.lt 0 a
      x : Real
      ⊢ ∀ (y : Prod EReal EReal), Membership.mem (SProd.sprod (Set.Ioi ↑(HDiv.hDiv ( …
    -/
  · intros p p_in_prod
    /-
      case h.refine_1
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p_in_prod : Membership.mem (SProd.sprod (Set.Ioi ↑(HDiv.hDiv (HMul.hMul 2 (Max …
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    simp only [Set.mem_prod, Set.mem_Ioi] at p_in_prod
    /-
      case h.refine_1
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p_in_prod : And (LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0))  …
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    rcases p_in_prod with ⟨p1_gt, p2_gt⟩
    have p1_pos : 0 < p.1 := by
      apply lt_of_le_of_lt _ p1_gt
      rw [EReal.coe_nonneg]
      apply mul_nonneg _ (le_of_lt (inv_pos_of_pos h))
      simp only [gt_iff_lt, Nat.ofNat_pos, mul_nonneg_iff_of_pos_left, le_max_iff, le_refl, or_true]
    /-
      case h.refine_1.intro
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p1_gt : LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0)) a)) p.1
      p2_gt : LT.lt (↑(HDiv.hDiv a 2)) p.2
      p1_pos : LT.lt 0 p.1
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    have a2_pos : 0 < ((a/2 : ℝ) : EReal) := by rw [EReal.coe_pos]; linarith
    /-
      case h.refine_1.intro
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p1_gt : LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0)) a)) p.1
      p2_gt : LT.lt (↑(HDiv.hDiv a 2)) p.2
      p1_pos : LT.lt 0 p.1
      a2_pos : LT.lt 0 ↑(HDiv.hDiv a 2)
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    have lock := mul_le_mul_of_nonneg_right (le_of_lt p1_gt) (le_of_lt a2_pos)
    /-
      case h.refine_1.intro
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p1_gt : LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0)) a)) p.1
      p2_gt : LT.lt (↑(HDiv.hDiv a 2)) p.2
      p1_pos : LT.lt 0 p.1
      a2_pos : LT.lt 0 ↑(HDiv.hDiv a 2)
      lock : LE.le (HMul.hMul ↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0))  …
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    have key := mul_le_mul_of_nonneg_left (le_of_lt p2_gt) (le_of_lt p1_pos)
    /-
      case h.refine_1.intro
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p1_gt : LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0)) a)) p.1
      p2_gt : LT.lt (↑(HDiv.hDiv a 2)) p.2
      p1_pos : LT.lt 0 p.1
      a2_pos : LT.lt 0 ↑(HDiv.hDiv a 2)
      lock : LE.le (HMul.hMul ↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0))  …
      key : LE.le (HMul.hMul p.1 ↑(HDiv.hDiv a 2)) (HMul.hMul p.1 p.2)
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    replace lock := le_trans lock key
    /-
      case h.refine_1.intro
      a : Real
      h : LT.lt 0 a
      x : Real
      p : Prod EReal EReal
      p1_gt : LT.lt (↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0)) a)) p.1
      p2_gt : LT.lt (↑(HDiv.hDiv a 2)) p.2
      p1_pos : LT.lt 0 p.1
      a2_pos : LT.lt 0 ↑(HDiv.hDiv a 2)
      key : LE.le (HMul.hMul p.1 ↑(HDiv.hDiv a 2)) (HMul.hMul p.1 p.2)
      lock : LE.le (HMul.hMul ↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0))  …
      ⊢ LT.lt (↑x) (HMul.hMul p.1 p.2)
    -/
    apply lt_of_lt_of_le _ lock
    rw [← EReal.coe_mul, EReal.coe_lt_coe_iff, div_mul_div_comm, mul_comm,
      ← div_mul_div_comm, mul_div_right_comm]
    simp only [ne_eq, Ne.symm (ne_of_lt h), not_false_eq_true, _root_.div_self, OfNat.ofNat_ne_zero,
      one_mul, lt_max_iff, lt_add_iff_pos_right, zero_lt_one, true_or]
    /-
      case h.refine_2.refine_1
      a : Real
      h : LT.lt 0 a
      x : Real
      ⊢ IsOpen (SProd.sprod (Set.Ioi ↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x  …
    -/
  · exact IsOpen.prod isOpen_Ioi isOpen_Ioi
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.refine_2.refine_1
      a : Real
      h : LT.lt 0 a
      x : Real
      ⊢ Membership.mem (Set.Ioi ↑(HDiv.hDiv (HMul.hMul 2 (Max.max (HAdd.hAdd x 1) 0) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.refine_2.refine_2
      a : Real
      h : LT.lt 0 a
      x : Real
      ⊢ Membership.mem (Set.Ioi ↑(HDiv.hDiv a 2)) { fst := Top.top, snd := ↑a }.2
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


private lemma continuousAt_mul_top_ne_zero {a : ℝ} (h : a ≠ 0) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) (⊤, a) := by
  /-
    a : Real
    h : Ne a 0
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a }
  -/
  rcases lt_or_gt_of_ne h with a_neg | a_pos
    /-
      case inl
      a : Real
      h : Ne a 0
      a_neg : LT.lt a 0
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a }
    -/
  · exact neg_neg a ▸ continuousAt_mul_symm2 (continuousAt_mul_top_pos (neg_pos.2 a_neg))
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Real
      h : Ne a 0
      a_pos : GT.gt a 0
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a }
    -/
  · exact continuousAt_mul_top_pos a_pos
    /-
      🎉 no goals
    -/


/-- The multiplication on `EReal` is continuous except at indeterminacies
(i.e. whenever one value is zero and the other infinite). -/
theorem continuousAt_mul {p : EReal × EReal} (h₁ : p.1 ≠ 0 ∨ p.2 ≠ ⊥)
    (h₂ : p.1 ≠ 0 ∨ p.2 ≠ ⊤) (h₃ : p.1 ≠ ⊥ ∨ p.2 ≠ 0) (h₄ : p.1 ≠ ⊤ ∨ p.2 ≠ 0) :
    ContinuousAt (fun p : EReal × EReal ↦ p.1 * p.2) p := by
  /-
    p : Prod EReal EReal
    h₁ : Or (Ne p.1 0) (Ne p.2 Bot.bot)
    h₂ : Or (Ne p.1 0) (Ne p.2 Top.top)
    h₃ : Or (Ne p.1 Bot.bot) (Ne p.2 0)
    h₄ : Or (Ne p.1 Top.top) (Ne p.2 0)
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) p
  -/
  rcases p with ⟨x, y⟩
  /-
    case mk
    x y : EReal
    h₁ : Or (Ne { fst := x, snd := y }.1 0) (Ne { fst := x, snd := y }.2 Bot.bot)
    h₂ : Or (Ne { fst := x, snd := y }.1 0) (Ne { fst := x, snd := y }.2 Top.top)
    h₃ : Or (Ne { fst := x, snd := y }.1 Bot.bot) (Ne { fst := x, snd := y }.2 0)
    h₄ : Or (Ne { fst := x, snd := y }.1 Top.top) (Ne { fst := x, snd := y }.2 0)
    ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := x, snd := y }
  -/
  induction x <;> induction y
    /-
      case mk.h_bot.h_bot
      h₁ : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 0) (Ne { fst := Bot.bot, snd  …
      h₂ : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 0) (Ne { fst := Bot.bot, snd  …
      h₃ : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 Bot.bot) (Ne { fst := Bot.bot …
      h₄ : Or (Ne { fst := Bot.bot, snd := Bot.bot }.1 Top.top) (Ne { fst := Bot.bot …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Bot.bot, snd := Bot.bot }
    -/
  · exact continuousAt_mul_symm3 continuousAt_mul_top_top
    /-
      🎉 no goals
    -/
    /-
      case mk.h_bot.h_real
      a✝ : Real
      h₁ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 0) (Ne { fst := Bot.bot, snd := ↑ …
      h₂ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 0) (Ne { fst := Bot.bot, snd := ↑ …
      h₃ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := Bot.bot, sn …
      h₄ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 Top.top) (Ne { fst := Bot.bot, sn …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Bot.bot, snd := ↑a✝ }
    -/
  · simp only [ne_eq, not_true_eq_false, EReal.coe_eq_zero, false_or] at h₃
    /-
      case mk.h_bot.h_real
      a✝ : Real
      h₁ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 0) (Ne { fst := Bot.bot, snd := ↑ …
      h₂ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 0) (Ne { fst := Bot.bot, snd := ↑ …
      h₄ : Or (Ne { fst := Bot.bot, snd := ↑a✝ }.1 Top.top) (Ne { fst := Bot.bot, sn …
      h₃ : Not (Eq a✝ 0)
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Bot.bot, snd := ↑a✝ }
    -/
    exact continuousAt_mul_symm1 (continuousAt_mul_top_ne_zero h₃)
    /-
      🎉 no goals
    -/
    /-
      case mk.h_bot.h_top
      h₁ : Or (Ne { fst := Bot.bot, snd := Top.top }.1 0) (Ne { fst := Bot.bot, snd  …
      h₂ : Or (Ne { fst := Bot.bot, snd := Top.top }.1 0) (Ne { fst := Bot.bot, snd  …
      h₃ : Or (Ne { fst := Bot.bot, snd := Top.top }.1 Bot.bot) (Ne { fst := Bot.bot …
      h₄ : Or (Ne { fst := Bot.bot, snd := Top.top }.1 Top.top) (Ne { fst := Bot.bot …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Bot.bot, snd := Top.top }
    -/
  · exact EReal.neg_top ▸ continuousAt_mul_symm1 continuousAt_mul_top_top
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_bot
      a✝ : Real
      h₁ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 0) (Ne { fst := ↑a✝, snd := Bot.b …
      h₂ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 0) (Ne { fst := ↑a✝, snd := Bot.b …
      h₃ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      h₄ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Top.top) (Ne { fst := ↑a✝, snd := …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a✝, snd := Bot.bot }
    -/
  · simp only [ne_eq, EReal.coe_eq_zero, not_true_eq_false, or_false] at h₁
    /-
      case mk.h_real.h_bot
      a✝ : Real
      h₂ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 0) (Ne { fst := ↑a✝, snd := Bot.b …
      h₃ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      h₄ : Or (Ne { fst := ↑a✝, snd := Bot.bot }.1 Top.top) (Ne { fst := ↑a✝, snd := …
      h₁ : Not (Eq a✝ 0)
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a✝, snd := Bot.bot }
    -/
    exact continuousAt_mul_symm2 (continuousAt_mul_swap (continuousAt_mul_top_ne_zero h₁))
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_real
      a✝¹ a✝ : Real
      h₁ : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 0) (Ne { fst := ↑a✝¹, snd := ↑a✝ }.2 …
      h₂ : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 0) (Ne { fst := ↑a✝¹, snd := ↑a✝ }.2 …
      h₃ : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := ↑a✝¹, snd := ↑ …
      h₄ : Or (Ne { fst := ↑a✝¹, snd := ↑a✝ }.1 Top.top) (Ne { fst := ↑a✝¹, snd := ↑ …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a✝¹, snd := ↑a✝ }
    -/
  · exact continuousAt_mul_coe_coe _ _
    /-
      🎉 no goals
    -/
    /-
      case mk.h_real.h_top
      a✝ : Real
      h₁ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 0) (Ne { fst := ↑a✝, snd := Top.t …
      h₂ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 0) (Ne { fst := ↑a✝, snd := Top.t …
      h₃ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      h₄ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Top.top) (Ne { fst := ↑a✝, snd := …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a✝, snd := Top.top }
    -/
  · simp only [ne_eq, EReal.coe_eq_zero, not_true_eq_false, or_false] at h₂
    /-
      case mk.h_real.h_top
      a✝ : Real
      h₁ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 0) (Ne { fst := ↑a✝, snd := Top.t …
      h₃ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Bot.bot) (Ne { fst := ↑a✝, snd := …
      h₄ : Or (Ne { fst := ↑a✝, snd := Top.top }.1 Top.top) (Ne { fst := ↑a✝, snd := …
      h₂ : Not (Eq a✝ 0)
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a✝, snd := Top.top }
    -/
    exact continuousAt_mul_swap (continuousAt_mul_top_ne_zero h₂)
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_bot
      h₁ : Or (Ne { fst := Top.top, snd := Bot.bot }.1 0) (Ne { fst := Top.top, snd  …
      h₂ : Or (Ne { fst := Top.top, snd := Bot.bot }.1 0) (Ne { fst := Top.top, snd  …
      h₃ : Or (Ne { fst := Top.top, snd := Bot.bot }.1 Bot.bot) (Ne { fst := Top.top …
      h₄ : Or (Ne { fst := Top.top, snd := Bot.bot }.1 Top.top) (Ne { fst := Top.top …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := Bot.bot }
    -/
  · exact continuousAt_mul_symm2 continuousAt_mul_top_top
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_real
      a✝ : Real
      h₁ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 0) (Ne { fst := Top.top, snd := ↑ …
      h₂ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 0) (Ne { fst := Top.top, snd := ↑ …
      h₃ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := Top.top, sn …
      h₄ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 Top.top) (Ne { fst := Top.top, sn …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a✝ }
    -/
  · simp only [ne_eq, not_true_eq_false, EReal.coe_eq_zero, false_or] at h₄
    /-
      case mk.h_top.h_real
      a✝ : Real
      h₁ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 0) (Ne { fst := Top.top, snd := ↑ …
      h₂ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 0) (Ne { fst := Top.top, snd := ↑ …
      h₃ : Or (Ne { fst := Top.top, snd := ↑a✝ }.1 Bot.bot) (Ne { fst := Top.top, sn …
      h₄ : Not (Eq a✝ 0)
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := ↑a✝ }
    -/
    exact continuousAt_mul_top_ne_zero h₄
    /-
      🎉 no goals
    -/
    /-
      case mk.h_top.h_top
      h₁ : Or (Ne { fst := Top.top, snd := Top.top }.1 0) (Ne { fst := Top.top, snd  …
      h₂ : Or (Ne { fst := Top.top, snd := Top.top }.1 0) (Ne { fst := Top.top, snd  …
      h₃ : Or (Ne { fst := Top.top, snd := Top.top }.1 Bot.bot) (Ne { fst := Top.top …
      h₄ : Or (Ne { fst := Top.top, snd := Top.top }.1 Top.top) (Ne { fst := Top.top …
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := Top.top, snd := Top.top }
    -/
  · exact continuousAt_mul_top_top
    /-
      🎉 no goals
    -/


lemma lowerSemicontinuous_add : LowerSemicontinuous fun p : EReal × EReal ↦ p.1 + p.2 := by
  /-
    ⊢ LowerSemicontinuous fun p => HAdd.hAdd p.1 p.2
  -/
  intro x y
  /-
    x : Prod EReal EReal
    y : EReal
    ⊢ LT.lt y ((fun p => HAdd.hAdd p.1 p.2) x) → Filter.Eventually (fun x' => LT.l …
  -/
  by_cases hx₁ : x.1 = ⊥
    /-
      case pos
      x : Prod EReal EReal
      y : EReal
      hx₁ : Eq x.1 Bot.bot
      ⊢ LT.lt y ((fun p => HAdd.hAdd p.1 p.2) x) → Filter.Eventually (fun x' => LT.l …
    -/
  · simp [hx₁]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Prod EReal EReal
    y : EReal
    hx₁ : Not (Eq x.1 Bot.bot)
    ⊢ LT.lt y ((fun p => HAdd.hAdd p.1 p.2) x) → Filter.Eventually (fun x' => LT.l …
  -/
  by_cases hx₂ : x.2 = ⊥
    /-
      case pos
      x : Prod EReal EReal
      y : EReal
      hx₁ : Not (Eq x.1 Bot.bot)
      hx₂ : Eq x.2 Bot.bot
      ⊢ LT.lt y ((fun p => HAdd.hAdd p.1 p.2) x) → Filter.Eventually (fun x' => LT.l …
    -/
  · simp [hx₂]
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Prod EReal EReal
      y : EReal
      hx₁ : Not (Eq x.1 Bot.bot)
      hx₂ : Not (Eq x.2 Bot.bot)
      ⊢ LT.lt y ((fun p => HAdd.hAdd p.1 p.2) x) → Filter.Eventually (fun x' => LT.l …
    -/
  · exact continuousAt_add (.inr hx₂) (.inl hx₁) |>.lowerSemicontinuousAt _
    /-
      🎉 no goals
    -/



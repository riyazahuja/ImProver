/-- The relation we induct along in the proof by DeVos of the Cauchy-Davenport theorem.
`(s₁, t₁) < (s₂, t₂)` iff
* `|s₁ * t₁| < |s₂ * t₂|`
* or `|s₁ * t₁| = |s₂ * t₂|` and `|s₂| + |t₂| < |s₁| + |t₁|`
* or `|s₁ * t₁| = |s₂ * t₂|` and `|s₁| + |t₁| = |s₂| + |t₂|` and `|s₁| < |s₂|`. -/
@[to_additive
"The relation we induct along in the proof by DeVos of the Cauchy-Davenport theorem.
`(s₁, t₁) < (s₂, t₂)` iff
* `|s₁ + t₁| < |s₂ + t₂|`
* or `|s₁ + t₁| = |s₂ + t₂|` and `|s₂| + |t₂| < |s₁| + |t₁|`
* or `|s₁ + t₁| = |s₂ + t₂|` and `|s₁| + |t₁| = |s₂| + |t₂|` and `|s₁| < |s₂|`."]
private def DevosMulRel : Finset α × Finset α → Finset α × Finset α → Prop :=
  Prod.Lex (· < ·) (Prod.Lex (· > ·) (· < ·)) on fun x ↦ (#(x.1 * x.2), #x.1 + #x.2, #x.1)


@[to_additive]
private lemma devosMulRel_iff :
    DevosMulRel x y ↔
      #(x.1 * x.2) < #(y.1 * y.2) ∨
        #(x.1 * x.2) = #(y.1 * y.2) ∧ #y.1 + #y.2 < #x.1 + #x.2 ∨
          #(x.1 * x.2) = #(y.1 * y.2) ∧ #x.1 + #x.2 = #y.1 + #y.2 ∧ #x.1 < #y.1 := by
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    x y : Prod (Finset α) (Finset α)
    ⊢ Iff (DevosMulRel x y) (Or (LT.lt (HMul.hMul x.1 x.2).card (HMul.hMul y.1 y.2 …
  -/
  simp [DevosMulRel, Prod.lex_iff, and_or_left]
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma devosMulRel_of_le (mul : #(x.1 * x.2) ≤ #(y.1 * y.2))
    (hadd : #y.1 + #y.2 < #x.1 + #x.2) : DevosMulRel x y :=
  devosMulRel_iff.2 <| mul.lt_or_eq.imp_right fun h ↦ Or.inl ⟨h, hadd⟩


@[to_additive]
private lemma devosMulRel_of_le_of_le (mul : #(x.1 * x.2) ≤ #(y.1 * y.2))
    (hadd : #y.1 + #y.2 ≤ #x.1 + #x.2) (hone : #x.1 < #y.1) : DevosMulRel x y :=
  devosMulRel_iff.2 <|
    mul.lt_or_eq.imp_right fun h ↦ hadd.gt_or_eq.imp (And.intro h) fun h' ↦ ⟨h, h', hone⟩


@[to_additive]
private lemma wellFoundedOn_devosMulRel :
    {x : Finset α × Finset α | x.1.Nonempty ∧ x.2.Nonempty}.WellFoundedOn
      (DevosMulRel : Finset α × Finset α → Finset α × Finset α → Prop) := by
  refine wellFounded_lt.onFun.wellFoundedOn.prod_lex_of_wellFoundedOn_fiber fun n ↦
    Set.WellFoundedOn.prod_lex_of_wellFoundedOn_fiber ?_ fun n ↦
      wellFounded_lt.onFun.wellFoundedOn
  exact wellFounded_lt.onFun.wellFoundedOn.mono' fun x hx y _ ↦ tsub_lt_tsub_left_of_le <|
    add_le_add ((card_le_card_mul_right hx.1.2).trans_eq hx.2) <|
      (card_le_card_mul_left hx.1.1).trans_eq hx.2


/-- A generalisation of the **Cauchy-Davenport theorem** to arbitrary groups. The size of `s * t` is
lower-bounded by `|s| + |t| - 1` unless this quantity is greater than the size of the smallest
subgroup. -/
@[to_additive "A generalisation of the **Cauchy-Davenport theorem** to arbitrary groups. The size of
`s + t` is lower-bounded by `|s| + |t| - 1` unless this quantity is greater than the size of the
smallest subgroup."]
lemma cauchy_davenport_minOrder_mul (hs : s.Nonempty) (ht : t.Nonempty) :
    min (minOrder α) ↑(#s + #t - 1) ≤ #(s * t) := by
  -- Set up the induction on `x := (s, t)` along the `DevosMulRel` relation.
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ LE.le (Min.min (Monoid.minOrder α) ↑(HSub.hSub (HAdd.hAdd s.card t.card) 1)) …
  -/
  set x := (s, t) with hx
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    x : Prod (Finset α) (Finset α) := { fst := s, snd := t }
    hx : Eq x { fst := s, snd := t }
    ⊢ LE.le (Min.min (Monoid.minOrder α) ↑(HSub.hSub (HAdd.hAdd s.card t.card) 1)) …
  -/
  clear_value x
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    x : Prod (Finset α) (Finset α)
    hx : Eq x { fst := s, snd := t }
    ⊢ LE.le (Min.min (Monoid.minOrder α) ↑(HSub.hSub (HAdd.hAdd s.card t.card) 1)) …
  -/
  simp only [Prod.ext_iff] at hx
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    x : Prod (Finset α) (Finset α)
    hx : And (Eq x.1 s) (Eq x.2 t)
    ⊢ LE.le (Min.min (Monoid.minOrder α) ↑(HSub.hSub (HAdd.hAdd s.card t.card) 1)) …
  -/
  obtain ⟨rfl, rfl⟩ := hx
  refine wellFoundedOn_devosMulRel.induction (P := fun x : Finset α × Finset α ↦
    min (minOrder α) ↑(#x.1 + #x.2 - 1) ≤ #(x.1 * x.2)) ⟨hs, ht⟩ ?_
  /-
    case intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    x : Prod (Finset α) (Finset α)
    hs : x.1.Nonempty
    ht : x.2.Nonempty
    ⊢ ∀ (y : Prod (Finset α) (Finset α)), Membership.mem (setOf fun x => And x.1.N …
  -/
  clear! x
  /-
    case intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    ⊢ ∀ (y : Prod (Finset α) (Finset α)), Membership.mem (setOf fun x => And x.1.N …
  -/
  rintro ⟨s, t⟩ ⟨hs, ht⟩ ih
  simp only [min_le_iff, tsub_le_iff_right, Prod.forall, Set.mem_setOf_eq, and_imp,
    Nat.cast_le] at *
  -- If `#t < #s`, we're done by the induction hypothesis on `(t⁻¹, s⁻¹)`.
  /-
    case intro.mk.intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  obtain hts | hst := lt_or_le #t #s
  · simpa only [← mul_inv_rev, add_comm, card_inv] using
      ih _ _ ht.inv hs.inv
        (devosMulRel_iff.2 <| Or.inr <| Or.inr <| by
          simpa only [← mul_inv_rev, add_comm, card_inv, true_and])
  -- If `s` is a singleton, then the result is trivial.
  /-
    case intro.mk.intro.inr
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  obtain ⟨a, rfl⟩ | ⟨a, ha, b, hb, hab⟩ := hs.exists_eq_singleton_or_nontrivial
    /-
      case intro.mk.intro.inr.inl.intro
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      t : Finset α
      ht : t.Nonempty
      a : α
      hs : (Singleton.singleton a).Nonempty
      ih : ∀ (a_1 b : Finset α), a_1.Nonempty → b.Nonempty → DevosMulRel { fst := a_ …
      hst : LE.le (Singleton.singleton a).card t.card
      ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul (Singleton.singleton a) t).card) ( …
    -/
  · simp [add_comm]
    /-
      🎉 no goals
    -/
  -- Else, we have `a, b ∈ s` distinct. So `g := b⁻¹ * a` is a non-identity element such that `s`
  -- intersects its right translate by `g`.
  obtain ⟨g, hg, hgs⟩ : ∃ g : α, g ≠ 1 ∧ (s ∩ op g • s).Nonempty :=
    ⟨b⁻¹ * a, inv_mul_eq_one.not.2 hab.symm, _,
      mem_inter.2 ⟨ha, mem_smul_finset.2 ⟨_, hb, by simp⟩⟩⟩
  -- If `s` is equal to its right translate by `g`, then it contains a nontrivial subgroup, namely
  -- the subgroup generated by `g`. So `s * t` has size at least the size of a nontrivial subgroup,
  -- as wanted.
  /-
    case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    a : α
    ha : Membership.mem (↑s) a
    b : α
    hb : Membership.mem (↑s) b
    hab : Ne a b
    g : α
    hg : Ne g 1
    hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  obtain hsg | hsg := eq_or_ne (op g • s) s
  · have hS : (zpowers g : Set α) ⊆ a⁻¹ • (s : Set α) := by
      refine forall_mem_zpowers.2 <| @zpow_induction_right _ _ _ (· ∈ a⁻¹ • (s : Set α))
        ⟨_, ha, inv_mul_cancel _⟩ (fun c hc ↦ ?_) fun c hc ↦ ?_
      · rw [← hsg, coe_smul_finset, smul_comm]
        exact Set.smul_mem_smul_set hc
      · simp only
        rwa [← op_smul_eq_mul, op_inv, ← Set.mem_smul_set_iff_inv_smul_mem, smul_comm,
          ← coe_smul_finset, hsg]
    refine Or.inl ((minOrder_le_natCard (zpowers_ne_bot.2 hg) <|
      s.finite_toSet.smul_set.subset hS).trans <| WithTop.coe_le_coe.2 <|
        ((Nat.card_mono s.finite_toSet.smul_set hS).trans_eq <| ?_).trans <|
          card_le_card_mul_right ht)
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : Eq (HSMul.hSMul (MulOpposite.op g) s) s
      hS : HasSubset.Subset (↑(Subgroup.zpowers g)) (HSMul.hSMul (Inv.inv a) ↑s)
      ⊢ Eq (Nat.card ↑(HSMul.hSMul (Inv.inv a) ↑s)) s.card
    -/
    rw [← coe_smul_finset]
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : Eq (HSMul.hSMul (MulOpposite.op g) s) s
      hS : HasSubset.Subset (↑(Subgroup.zpowers g)) (HSMul.hSMul (Inv.inv a) ↑s)
      ⊢ Eq (Nat.card ↑↑(HSMul.hSMul (Inv.inv a) s)) s.card
    -/
    simp [-coe_smul_finset]
    /-
      🎉 no goals
    -/
  -- Else, we can transform `s`, `t` to `s'`, `t'` and `s''`, `t''`, such that one of `(s', t')` and
  -- `(s'', t'')` is strictly smaller than `(s, t)` according to `DevosMulRel`.
  replace hsg : #(s ∩ op g • s) < #s := card_lt_card ⟨inter_subset_left, fun h ↦
    hsg <| eq_of_superset_of_card_ge (h.trans inter_subset_right) (card_smul_finset _ _).le⟩
  /-
    case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    a : α
    ha : Membership.mem (↑s) a
    b : α
    hb : Membership.mem (↑s) b
    hab : Ne a b
    g : α
    hg : Ne g 1
    hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
    hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  replace aux1 := card_mono <| mulETransformLeft.fst_mul_snd_subset g (s, t)
  /-
    case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    a : α
    ha : Membership.mem (↑s) a
    b : α
    hb : Membership.mem (↑s) b
    hab : Ne a b
    g : α
    hg : Ne g 1
    hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
    hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
    aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  replace aux2 := card_mono <| mulETransformRight.fst_mul_snd_subset g (s, t)
  -- If the left translate of `t` by `g⁻¹` is disjoint from `t`, then we're easily done.
  /-
    case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    a : α
    ha : Membership.mem (↑s) a
    b : α
    hb : Membership.mem (↑s) b
    hab : Ne a b
    g : α
    hg : Ne g 1
    hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
    hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
    aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
    aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  obtain hgt | hgt := disjoint_or_nonempty_inter t (g⁻¹ • t)
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
      aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
      aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
      hgt : Disjoint t (HSMul.hSMul (Inv.inv g) t)
      ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
    -/
  · rw [← card_smul_finset g⁻¹ t]
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
      aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
      aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
      hgt : Disjoint t (HSMul.hSMul (Inv.inv g) t)
      ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
    -/
    refine Or.inr ((add_le_add_right hst _).trans ?_)
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
      aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
      aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
      hgt : Disjoint t (HSMul.hSMul (Inv.inv g) t)
      ⊢ LE.le (HAdd.hAdd t.card (HSMul.hSMul (Inv.inv g) t).card) (HAdd.hAdd (HMul.h …
    -/
    rw [← card_union_of_disjoint hgt]
    /-
      case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝¹ : Group α
      inst✝ : DecidableEq α
      s t : Finset α
      hs : s.Nonempty
      ht : t.Nonempty
      ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
      hst : LE.le s.card t.card
      a : α
      ha : Membership.mem (↑s) a
      b : α
      hb : Membership.mem (↑s) b
      hab : Ne a b
      g : α
      hg : Ne g 1
      hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
      hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
      aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
      aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
      hgt : Disjoint t (HSMul.hSMul (Inv.inv g) t)
      ⊢ LE.le (Union.union t (HSMul.hSMul (Inv.inv g) t)).card (HAdd.hAdd (HMul.hMul …
    -/
    exact (card_le_card_mul_left hgs).trans (le_add_of_le_left aux1)
    /-
      🎉 no goals
    -/
  -- Else, we're done by induction on either `(s', t')` or `(s'', t'')` depending on whether
  -- `|s| + |t| ≤ |s'| + |t'|` or `|s| + |t| < |s''| + |t''|`. One of those two inequalities must
  -- hold since `2 * (|s| + |t|) = |s'| + |t'| + |s''| + |t''|`.
  /-
    case intro.mk.intro.inr.inr.intro.intro.intro.intro.intro.intro.inr.inr
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : DecidableEq α
    s t : Finset α
    hs : s.Nonempty
    ht : t.Nonempty
    ih : ∀ (a b : Finset α), a.Nonempty → b.Nonempty → DevosMulRel { fst := a, snd …
    hst : LE.le s.card t.card
    a : α
    ha : Membership.mem (↑s) a
    b : α
    hb : Membership.mem (↑s) b
    hab : Ne a b
    g : α
    hg : Ne g 1
    hgs : (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).Nonempty
    hsg : LT.lt (Inter.inter s (HSMul.hSMul (MulOpposite.op g) s)).card s.card
    aux1 : LE.le (HMul.hMul (Finset.mulETransformLeft g { fst := s, snd := t }).1  …
    aux2 : LE.le (HMul.hMul (Finset.mulETransformRight g { fst := s, snd := t }).1 …
    hgt : (Inter.inter t (HSMul.hSMul (Inv.inv g) t)).Nonempty
    ⊢ Or (LE.le (Monoid.minOrder α) ↑(HMul.hMul s t).card) (LE.le (HAdd.hAdd s.car …
  -/
  obtain hstg | hstg := le_or_lt_of_add_le_add (MulETransform.card g (s, t)).ge
  · exact (ih _ _ hgs (hgt.mono inter_subset_union) <| devosMulRel_of_le_of_le aux1 hstg hsg).imp
      (WithTop.coe_le_coe.2 aux1).trans' fun h ↦ hstg.trans <| h.trans <| add_le_add_right aux1 _
  · exact (ih _ _ (hgs.mono inter_subset_union) hgt <| devosMulRel_of_le aux2 hstg).imp
      (WithTop.coe_le_coe.2 aux2).trans' fun h ↦
        hstg.le.trans <| h.trans <| add_le_add_right aux2 _


/-- The **Cauchy-Davenport Theorem** for torsion-free groups. The size of `s * t` is lower-bounded
by `|s| + |t| - 1`. -/
@[to_additive
"The **Cauchy-Davenport theorem** for torsion-free groups. The size of `s + t` is lower-bounded
by `|s| + |t| - 1`."]
lemma cauchy_davenport_mul_of_isTorsionFree (h : IsTorsionFree α)
    (hs : s.Nonempty) (ht : t.Nonempty) : #s + #t - 1 ≤ #(s * t) := by
  simpa only [h.minOrder, min_eq_right, le_top, Nat.cast_le]
    using cauchy_davenport_minOrder_mul hs ht


/-- The **Cauchy-Davenport Theorem**. If `s`, `t` are nonempty sets in $$ℤ/pℤ$$, then the size of
`s + t` is lower-bounded by `|s| + |t| - 1`, unless this quantity is greater than `p`. -/
lemma ZMod.cauchy_davenport {p : ℕ} (hp : p.Prime) {s t : Finset (ZMod p)} (hs : s.Nonempty)
    (ht : t.Nonempty) : min p (#s + #t - 1) ≤ #(s + t) := by
  simpa only [ZMod.minOrder_of_prime hp, min_le_iff, Nat.cast_le]
    using cauchy_davenport_minOrder_add hs ht


/-- The **Cauchy-Davenport Theorem** for linearly ordered cancellative semigroups. The size of
`s * t` is lower-bounded by `|s| + |t| - 1`. -/
@[to_additive
"The **Cauchy-Davenport theorem** for linearly ordered additive cancellative semigroups. The size of
`s + t` is lower-bounded by `|s| + |t| - 1`."]
lemma cauchy_davenport_mul_of_linearOrder_isCancelMul [LinearOrder α] [Semigroup α] [IsCancelMul α]
    [MulLeftMono α] [MulRightMono α]
    {s t : Finset α} (hs : s.Nonempty) (ht : t.Nonempty) : #s + #t - 1 ≤ #(s * t) := by
  suffices s * {t.min' ht} ∩ ({s.max' hs} * t) = {s.max' hs * t.min' ht} by
    rw [← card_singleton_mul (s.max' hs) t, ← card_mul_singleton s (t.min' ht),
      ← card_union_add_card_inter, ← card_singleton _, ← this, Nat.add_sub_cancel]
    exact card_mono (union_subset (mul_subset_mul_left <| singleton_subset_iff.2 <| min'_mem _ _) <|
      mul_subset_mul_right <| singleton_subset_iff.2 <| max'_mem _ _)
  refine eq_singleton_iff_unique_mem.2 ⟨mem_inter.2 ⟨mul_mem_mul (max'_mem _ _) <|
    mem_singleton_self _, mul_mem_mul (mem_singleton_self _) <| min'_mem _ _⟩, ?_⟩
  simp only [mem_inter, and_imp, mem_mul, mem_singleton, exists_and_left, exists_eq_left,
    forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, mul_left_inj]
  exact fun a' ha' b' hb' h ↦ (le_max' _ _ ha').eq_of_not_lt fun ha ↦
    ((mul_lt_mul_right' ha _).trans_eq' h).not_le <| mul_le_mul_left' (min'_le _ _ hb') _


/-- The subfield of F fixed by the field endomorphism `m`. -/
def FixedBy.subfield : Subfield F where
  carrier := fixedBy F m
  zero_mem' := smul_zero m
  add_mem' hx hy := (smul_add m _ _).trans <| congr_arg₂ _ hx hy
  neg_mem' hx := (smul_neg m _).trans <| congr_arg _ hx
  one_mem' := smul_one m
  mul_mem' hx hy := (smul_mul' m _ _).trans <| congr_arg₂ _ hx hy
  inv_mem' x hx := (smul_inv'' m x).trans <| congr_arg _ hx


/-- A typeclass for subrings invariant under a `MulSemiringAction`. -/
class IsInvariantSubfield (S : Subfield F) : Prop where
  smul_mem : ∀ (m : M) {x : F}, x ∈ S → m • x ∈ S


instance IsInvariantSubfield.toMulSemiringAction [IsInvariantSubfield M S] :
    MulSemiringAction M S where
  smul m x := ⟨m • x.1, IsInvariantSubfield.smul_mem m x.2⟩
  one_smul s := Subtype.eq <| one_smul M s.1
  mul_smul m₁ m₂ s := Subtype.eq <| mul_smul m₁ m₂ s.1
  smul_add m s₁ s₂ := Subtype.eq <| smul_add m s₁.1 s₂.1
  smul_zero m := Subtype.eq <| smul_zero m
  smul_one m := Subtype.eq <| smul_one m
  smul_mul m s₁ s₂ := Subtype.eq <| smul_mul' m s₁.1 s₂.1


instance [IsInvariantSubfield M S] : IsInvariantSubring M S.toSubring where
  smul_mem := IsInvariantSubfield.smul_mem


/-- The subfield of fixed points by a monoid action. -/
def subfield : Subfield F :=
  Subfield.copy (⨅ m : M, FixedBy.subfield F m) (fixedPoints M F)
        /-
          M : Type u
          inst✝⁴ : Monoid M
          G : Type u
          inst✝³ : Group G
          F : Type v
          inst✝² : Field F
          inst✝¹ : MulSemiringAction M F
          inst✝ : MulSemiringAction G F
          m : M
          ⊢ Eq (MulAction.fixedPoints M F) ↑(iInf fun m => FixedBy.subfield F m)
        -/
    (by ext z; simp [fixedPoints, FixedBy.subfield, iInf, Subfield.mem_sInf]; rfl)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance : IsInvariantSubfield M (FixedPoints.subfield M F) where
                           /-
                             M : Type u
                             inst✝⁴ : Monoid M
                             G : Type u
                             inst✝³ : Group G
                             F : Type v
                             inst✝² : Field F
                             inst✝¹ : MulSemiringAction M F
                             inst✝ : MulSemiringAction G F
                             m g : M
                             x : F
                             hx : Membership.mem (FixedPoints.subfield M F) x
                             g' : M
                             ⊢ Eq (HSMul.hSMul g' (HSMul.hSMul g x)) (HSMul.hSMul g x)
                           -/
  smul_mem g x hx g' := by rw [hx, hx]
                           /-
                             🎉 no goals
                           -/


instance : SMulCommClass M (FixedPoints.subfield M F) F where
                                                         /-
                                                           M : Type u
                                                           inst✝⁴ : Monoid M
                                                           G : Type u
                                                           inst✝³ : Group G
                                                           F : Type v
                                                           inst✝² : Field F
                                                           inst✝¹ : MulSemiringAction M F
                                                           inst✝ : MulSemiringAction G F
                                                           m✝ m : M
                                                           f : Subtype fun x => Membership.mem (FixedPoints.subfield M F) x
                                                           f' : F
                                                           ⊢ Eq (HSMul.hSMul m (HMul.hMul (↑f) f')) (HMul.hMul (↑f) (HSMul.hSMul m f'))
                                                         -/
  smul_comm m f f' := show m • (↑f * f') = f * m • f' by rw [smul_mul', f.prop m]
                                                         /-
                                                           🎉 no goals
                                                         -/


instance smulCommClass' : SMulCommClass (FixedPoints.subfield M F) M F :=
  SMulCommClass.symm _ _ _


@[simp]
theorem smul (m : M) (x : FixedPoints.subfield M F) : m • x = x :=
  Subtype.eq <| x.2 m

-- Why is this so slow?

@[simp]
theorem smul_polynomial (m : M) (p : Polynomial (FixedPoints.subfield M F)) : m • p = p :=
                                         /-
                                           M : Type u
                                           inst✝² : Monoid M
                                           F : Type v
                                           inst✝¹ : Field F
                                           inst✝ : MulSemiringAction M F
                                           m : M
                                           p : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield M F) x)
                                           x : Subtype fun x => Membership.mem (FixedPoints.subfield M F) x
                                           ⊢ Eq (HSMul.hSMul m (Polynomial.C x)) (Polynomial.C x)
                                         -/
  Polynomial.induction_on p (fun x => by rw [Polynomial.smul_C, smul])
                                         /-
                                           🎉 no goals
                                         -/
                           /-
                             M : Type u
                             inst✝² : Monoid M
                             F : Type v
                             inst✝¹ : Field F
                             inst✝ : MulSemiringAction M F
                             m : M
                             p✝ p q : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield M F …
                             ihp : Eq (HSMul.hSMul m p) p
                             ihq : Eq (HSMul.hSMul m q) q
                             ⊢ Eq (HSMul.hSMul m (HAdd.hAdd p q)) (HAdd.hAdd p q)
                           -/
    (fun p q ihp ihq => by rw [smul_add, ihp, ihq]) fun n x _ => by
                           /-
                             🎉 no goals
                           -/
    /-
      M : Type u
      inst✝² : Monoid M
      F : Type v
      inst✝¹ : Field F
      inst✝ : MulSemiringAction M F
      m : M
      p : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield M F) x)
      n : Nat
      x : Subtype fun x => Membership.mem (FixedPoints.subfield M F) x
      x✝ : Eq (HSMul.hSMul m (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X n)) …
      ⊢ Eq (HSMul.hSMul m (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X (HAdd. …
    -/
    rw [smul_mul', Polynomial.smul_C, smul, smul_pow', Polynomial.smul_X]
    /-
      🎉 no goals
    -/


                                                      /-
                                                        M : Type u
                                                        inst✝⁴ : Monoid M
                                                        G : Type u
                                                        inst✝³ : Group G
                                                        F : Type v
                                                        inst✝² : Field F
                                                        inst✝¹ : MulSemiringAction M F
                                                        inst✝ : MulSemiringAction G F
                                                        m : M
                                                        ⊢ Algebra (Subtype fun x => Membership.mem (FixedPoints.subfield M F) x) F
                                                      -/
instance : Algebra (FixedPoints.subfield M F) F := by infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem coe_algebraMap :
    algebraMap (FixedPoints.subfield M F) F = Subfield.subtype (FixedPoints.subfield M F) :=
  rfl


theorem linearIndependent_smul_of_linearIndependent {s : Finset F} :
    (LinearIndependent (FixedPoints.subfield G F) fun i : (s : Set F) => (i : F)) →
      LinearIndependent F fun i : (s : Set F) => MulAction.toFun G F i := by
  classical
  have : IsEmpty ((∅ : Finset F) : Set F) := by simp
  refine Finset.induction_on s (fun _ => linearIndependent_empty_type) fun a s has ih hs => ?_
  rw [coe_insert] at hs ⊢
  rw [linearIndependent_insert (mt mem_coe.1 has)] at hs
  rw [linearIndependent_insert' (mt mem_coe.1 has)]; refine ⟨ih hs.1, fun ha => ?_⟩
  rw [Finsupp.mem_span_image_iff_linearCombination] at ha; rcases ha with ⟨l, hl, hla⟩
  rw [Finsupp.linearCombination_apply_of_mem_supported F hl] at hla
  suffices ∀ i ∈ s, l i ∈ FixedPoints.subfield G F by
    replace hla := (sum_apply _ _ fun i => l i • toFun G F i).symm.trans (congr_fun hla 1)
    simp_rw [Pi.smul_apply, toFun_apply, one_smul] at hla
    refine hs.2 (hla ▸ Submodule.sum_mem _ fun c hcs => ?_)
    change (⟨l c, this c hcs⟩ : FixedPoints.subfield G F) • c ∈ _
    exact Submodule.smul_mem _ _ (Submodule.subset_span <| mem_coe.2 hcs)
  intro i his g
  refine
    eq_of_sub_eq_zero
      (linearIndependent_iff'.1 (ih hs.1) s.attach (fun i => g • l i - l i) ?_ ⟨i, his⟩
          (mem_attach _ _) :
        _)
  refine (sum_attach s fun i ↦ (g • l i - l i) • MulAction.toFun G F i).trans ?_
  ext g'; dsimp only
  conv_lhs =>
    rw [sum_apply]
    congr
    · skip
    · ext
      rw [Pi.smul_apply, sub_smul, smul_eq_mul]
  rw [sum_sub_distrib, Pi.zero_apply, sub_eq_zero]
  conv_lhs =>
    congr
    · skip
    · ext x
      rw [toFun_apply, ← mul_inv_cancel_left g g', mul_smul, ← smul_mul', ← toFun_apply _ x]
  show
    (∑ x ∈ s, g • (fun y => l y • MulAction.toFun G F y) x (g⁻¹ * g')) =
      ∑ x ∈ s, (fun y => l y • MulAction.toFun G F y) x g'
  rw [← smul_sum, ← sum_apply _ _ fun y => l y • toFun G F y, ←
    sum_apply _ _ fun y => l y • toFun G F y]
  rw [hla, toFun_apply, toFun_apply, smul_smul, mul_inv_cancel_left]


/-- `minpoly G F x` is the minimal polynomial of `(x : F)` over `FixedPoints.subfield G F`. -/
def minpoly : Polynomial (FixedPoints.subfield G F) :=
  (prodXSubSMul G F x).toSubring (FixedPoints.subfield G F).toSubring fun _ hc g =>
    let ⟨n, _, hn⟩ := Polynomial.mem_coeffs_iff.1 hc
    hn.symm ▸ prodXSubSMul.coeff G F x g n


theorem monic : (minpoly G F x).Monic := by
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    ⊢ (FixedPoints.minpoly G F x).Monic
  -/
  simp only [minpoly]
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    ⊢ ((prodXSubSMul G F x).toSubring (FixedPoints.subfield G F).toSubring ⋯).Monic
  -/
  rw [Polynomial.monic_toSubring]
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    ⊢ (prodXSubSMul G F x).Monic
  -/
  exact prodXSubSMul.monic G F x
  /-
    🎉 no goals
  -/


theorem eval₂ :
    Polynomial.eval₂ (Subring.subtype <| (FixedPoints.subfield G F).toSubring) x (minpoly G F x) =
      0 := by
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    ⊢ Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x (FixedPoints.minpo …
  -/
  rw [← prodXSubSMul.eval G F x, Polynomial.eval₂_eq_eval_map]
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    ⊢ Eq (Polynomial.eval x (Polynomial.map (FixedPoints.subfield G F).subtype (Fi …
  -/
  simp only [minpoly, Polynomial.map_toSubring]
  /-
    🎉 no goals
  -/


theorem eval₂' :
    Polynomial.eval₂ (Subfield.subtype <| FixedPoints.subfield G F) x (minpoly G F x) = 0 :=
  eval₂ G F x


theorem ne_one : minpoly G F x ≠ (1 : Polynomial (FixedPoints.subfield G F)) := fun H =>
  have := eval₂ G F x
                                    /-
                                      G : Type u
                                      inst✝³ : Group G
                                      F : Type v
                                      inst✝² : Field F
                                      inst✝¹ : MulSemiringAction G F
                                      inst✝ : Fintype G
                                      x : F
                                      H : Eq (FixedPoints.minpoly G F x) 1
                                      this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x (FixedPoints. …
                                      ⊢ Eq 1 0
                                    -/
  (one_ne_zero : (1 : F) ≠ 0) <| by rwa [H, Polynomial.eval₂_one] at this
                                    /-
                                      🎉 no goals
                                    -/


theorem of_eval₂ (f : Polynomial (FixedPoints.subfield G F))
    (hf : Polynomial.eval₂ (Subfield.subtype <| FixedPoints.subfield G F) x f = 0) :
    minpoly G F x ∣ f := by
  classical
-- Porting note: the two `have` below were not needed.
  have : (subfield G F).subtype = (subfield G F).toSubring.subtype := rfl
  have h : Polynomial.map (MulSemiringActionHom.toRingHom (IsInvariantSubring.subtypeHom G
    (subfield G F).toSubring)) f = Polynomial.map
    ((IsInvariantSubring.subtypeHom G (subfield G F).toSubring)) f := rfl
  rw [← Polynomial.map_dvd_map' (Subfield.subtype <| FixedPoints.subfield G F), minpoly, this,
    Polynomial.map_toSubring _ _, prodXSubSMul]
  refine
    Fintype.prod_dvd_of_coprime
      (Polynomial.pairwise_coprime_X_sub_C <| MulAction.injective_ofQuotientStabilizer G x) fun y =>
      QuotientGroup.induction_on y fun g => ?_
  rw [Polynomial.dvd_iff_isRoot, Polynomial.IsRoot.def, MulAction.ofQuotientStabilizer_mk,
    Polynomial.eval_smul', ← this, ← Subfield.toSubring_subtype_eq_subtype, ←
    IsInvariantSubring.coe_subtypeHom' G (FixedPoints.subfield G F).toSubring, h,
    ← MulSemiringActionHom.coe_polynomial, ← MulSemiringActionHom.map_smul, smul_polynomial,
    MulSemiringActionHom.coe_polynomial, ← h, IsInvariantSubring.coe_subtypeHom',
    Polynomial.eval_map, Subfield.toSubring_subtype_eq_subtype, hf, smul_zero]

-- Why is this so slow?

theorem irreducible_aux (f g : Polynomial (FixedPoints.subfield G F)) (hf : f.Monic) (hg : g.Monic)
    (hfg : f * g = minpoly G F x) : f = 1 ∨ g = 1 := by
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
    hf : f.Monic
    hg : g.Monic
    hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  have hf2 : f ∣ minpoly G F x := by rw [← hfg]; exact dvd_mul_right _ _
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
    hf : f.Monic
    hg : g.Monic
    hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
    hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  have hg2 : g ∣ minpoly G F x := by rw [← hfg]; exact dvd_mul_left _ _
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
    hf : f.Monic
    hg : g.Monic
    hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
    hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
    hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  have := eval₂ G F x
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
    hf : f.Monic
    hg : g.Monic
    hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
    hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
    hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
    this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x (FixedPoints. …
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  rw [← hfg, Polynomial.eval₂_mul, mul_eq_zero] at this
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    x : F
    f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
    hf : f.Monic
    hg : g.Monic
    hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
    hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
    hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
    this : Or (Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x f) 0) (Eq …
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  cases' this with this this
    /-
      case inl
      G : Type u
      inst✝³ : Group G
      F : Type v
      inst✝² : Field F
      inst✝¹ : MulSemiringAction G F
      inst✝ : Fintype G
      x : F
      f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
      hf : f.Monic
      hg : g.Monic
      hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
      hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
      hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
      this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x f) 0
      ⊢ Or (Eq f 1) (Eq g 1)
    -/
  · right
    have hf3 : f = minpoly G F x :=
      Polynomial.eq_of_monic_of_associated hf (monic G F x)
        (associated_of_dvd_dvd hf2 <| @of_eval₂ G _ F _ _ _ x f this)
    /-
      case inl.h
      G : Type u
      inst✝³ : Group G
      F : Type v
      inst✝² : Field F
      inst✝¹ : MulSemiringAction G F
      inst✝ : Fintype G
      x : F
      f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
      hf : f.Monic
      hg : g.Monic
      hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
      hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
      hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
      this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x f) 0
      hf3 : Eq f (FixedPoints.minpoly G F x)
      ⊢ Eq g 1
    -/
    rwa [← mul_one (minpoly G F x), hf3, mul_right_inj' (monic G F x).ne_zero] at hfg
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u
      inst✝³ : Group G
      F : Type v
      inst✝² : Field F
      inst✝¹ : MulSemiringAction G F
      inst✝ : Fintype G
      x : F
      f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
      hf : f.Monic
      hg : g.Monic
      hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
      hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
      hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
      this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x g) 0
      ⊢ Or (Eq f 1) (Eq g 1)
    -/
  · left
    have hg3 : g = minpoly G F x :=
      Polynomial.eq_of_monic_of_associated hg (monic G F x)
        (associated_of_dvd_dvd hg2 <| @of_eval₂ G _ F _ _ _ x g this)
    /-
      case inr.h
      G : Type u
      inst✝³ : Group G
      F : Type v
      inst✝² : Field F
      inst✝¹ : MulSemiringAction G F
      inst✝ : Fintype G
      x : F
      f g : Polynomial (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x)
      hf : f.Monic
      hg : g.Monic
      hfg : Eq (HMul.hMul f g) (FixedPoints.minpoly G F x)
      hf2 : Dvd.dvd f (FixedPoints.minpoly G F x)
      hg2 : Dvd.dvd g (FixedPoints.minpoly G F x)
      this : Eq (Polynomial.eval₂ (FixedPoints.subfield G F).subtype x g) 0
      hg3 : Eq g (FixedPoints.minpoly G F x)
      ⊢ Eq f 1
    -/
    rwa [← one_mul (minpoly G F x), hg3, mul_left_inj' (monic G F x).ne_zero] at hfg
    /-
      🎉 no goals
    -/


theorem irreducible : Irreducible (minpoly G F x) :=
  (Polynomial.irreducible_of_monic (monic G F x) (ne_one G F x)).2 (irreducible_aux G F x)


theorem isIntegral [Finite G] (x : F) : IsIntegral (FixedPoints.subfield G F) x := by
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    x : F
    ⊢ IsIntegral (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x) x
  -/
  cases nonempty_fintype G; exact ⟨minpoly G F x, minpoly.monic G F x, minpoly.eval₂ G F x⟩
                            /-
                              🎉 no goals
                            -/


theorem minpoly_eq_minpoly : minpoly G F x = _root_.minpoly (FixedPoints.subfield G F) x :=
  minpoly.eq_of_irreducible_of_monic (minpoly.irreducible G F x) (minpoly.eval₂ G F x)
    (minpoly.monic G F x)


theorem rank_le_card : Module.rank (FixedPoints.subfield G F) F ≤ Fintype.card G :=
  rank_le fun s hs => by
    simpa only [rank_fun', Cardinal.mk_coe_finset, Finset.coe_sort_coe, Cardinal.lift_natCast,
      Nat.cast_le] using
      (linearIndependent_smul_of_linearIndependent G F hs).cardinal_lift_le_rank


instance normal : Normal (FixedPoints.subfield G F) F where
  isAlgebraic x := (isIntegral G F x).isAlgebraic
  splits' x :=
    (Polynomial.splits_id_iff_splits _).1 <| by
      /-
        M : Type u
        inst✝⁵ : Monoid M
        G : Type u
        inst✝⁴ : Group G
        F : Type v
        inst✝³ : Field F
        inst✝² : MulSemiringAction M F
        inst✝¹ : MulSemiringAction G F
        m : M
        inst✝ : Finite G
        x : F
        ⊢ Polynomial.Splits (RingHom.id F) (Polynomial.map (algebraMap (Subtype fun x  …
      -/
      cases nonempty_fintype G
      rw [← minpoly_eq_minpoly, minpoly, coe_algebraMap, ← Subfield.toSubring_subtype_eq_subtype,
        Polynomial.map_toSubring _ (subfield G F).toSubring, prodXSubSMul]
      /-
        case intro
        M : Type u
        inst✝⁵ : Monoid M
        G : Type u
        inst✝⁴ : Group G
        F : Type v
        inst✝³ : Field F
        inst✝² : MulSemiringAction M F
        inst✝¹ : MulSemiringAction G F
        m : M
        inst✝ : Finite G
        x : F
        val✝ : Fintype G
        ⊢ Polynomial.Splits (RingHom.id F) (Finset.univ.prod fun g => HSub.hSub Polyno …
      -/
      exact Polynomial.splits_prod _ fun _ _ => Polynomial.splits_X_sub_C _
      /-
        🎉 no goals
      -/


instance isSeparable : Algebra.IsSeparable (FixedPoints.subfield G F) F := by
  classical
  exact ⟨fun x => by
    cases nonempty_fintype G
    -- this was a plain rw when we were using unbundled subrings
    erw [IsSeparable, ← minpoly_eq_minpoly,
      ← Polynomial.separable_map (FixedPoints.subfield G F).subtype, minpoly,
      Polynomial.map_toSubring _ (subfield G F).toSubring]
    exact Polynomial.separable_prod_X_sub_C_iff.2 (injective_ofQuotientStabilizer G x)⟩


instance : FiniteDimensional (subfield G F) F := by
  /-
    M : Type u
    inst✝⁵ : Monoid M
    G : Type u
    inst✝⁴ : Group G
    F : Type v
    inst✝³ : Field F
    inst✝² : MulSemiringAction M F
    inst✝¹ : MulSemiringAction G F
    m : M
    inst✝ : Finite G
    ⊢ FiniteDimensional (Subtype fun x => Membership.mem (FixedPoints.subfield G F …
  -/
  cases nonempty_fintype G
  exact IsNoetherian.iff_fg.1
      (IsNoetherian.iff_rank_lt_aleph0.2 <| (rank_le_card G F).trans_lt <| Cardinal.nat_lt_aleph0 _)


theorem finrank_le_card [Fintype G] : finrank (subfield G F) F ≤ Fintype.card G := by
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    ⊢ LE.le (Module.finrank (Subtype fun x => Membership.mem (FixedPoints.subfield …
  -/
  rw [← @Nat.cast_le Cardinal, finrank_eq_rank]
  /-
    G : Type u
    inst✝³ : Group G
    F : Type v
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Fintype G
    ⊢ LE.le (Module.rank (Subtype fun x => Membership.mem (FixedPoints.subfield G  …
  -/
  apply rank_le_card
  /-
    🎉 no goals
  -/


theorem linearIndependent_toLinearMap (R : Type u) (A : Type v) (B : Type w) [CommSemiring R]
    [Ring A] [Algebra R A] [CommRing B] [IsDomain B] [Algebra R B] :
    LinearIndependent B (AlgHom.toLinearMap : (A →ₐ[R] B) → A →ₗ[R] B) :=
  have : LinearIndependent B (LinearMap.ltoFun R A B ∘ AlgHom.toLinearMap) :=
    ((linearIndependent_monoidHom A B).comp ((↑) : (A →ₐ[R] B) → A →* B) fun _ _ hfg =>
        AlgHom.ext fun _ => DFunLike.ext_iff.1 hfg _ :
      _)
  this.of_comp _


theorem cardinalMk_algHom (K : Type u) (V : Type v) (W : Type w) [Field K] [Field V] [Algebra K V]
    [FiniteDimensional K V] [Field W] [Algebra K W] :
    Cardinal.mk (V →ₐ[K] W) ≤ finrank W (V →ₗ[K] W) :=
  (linearIndependent_toLinearMap K V W).cardinalMk_le_finrank


@[deprecated (since := "2024-11-10")] alias cardinal_mk_algHom := cardinalMk_algHom


noncomputable instance AlgEquiv.fintype (K : Type u) (V : Type v) [Field K] [Field V] [Algebra K V]
    [FiniteDimensional K V] : Fintype (V ≃ₐ[K] V) :=
  Fintype.ofEquiv (V →ₐ[K] V) (algEquivEquivAlgHom K V).symm


theorem finrank_algHom (K : Type u) (V : Type v) [Field K] [Field V] [Algebra K V]
    [FiniteDimensional K V] : Fintype.card (V →ₐ[K] V) ≤ finrank V (V →ₗ[K] V) :=
  (linearIndependent_toLinearMap K V V).fintype_card_le_finrank


/-- Let $F$ be a field. Let $G$ be a finite group acting faithfully on $F$.
Then $[F : F^G] = |G|$. -/
@[stacks 09I3 "second part"]
theorem finrank_eq_card [Fintype G] [FaithfulSMul G F] :
    finrank (FixedPoints.subfield G F) F = Fintype.card G :=
  le_antisymm (FixedPoints.finrank_le_card G F) <|
    calc
      Fintype.card G ≤ Fintype.card (F →ₐ[FixedPoints.subfield G F] F) :=
        Fintype.card_le_of_injective _ (MulSemiringAction.toAlgHom_injective _ F)
      _ ≤ finrank F (F →ₗ[FixedPoints.subfield G F] F) := finrank_algHom (subfield G F) F
      _ = finrank (FixedPoints.subfield G F) F := finrank_linearMap_self _ _ _


/-- `MulSemiringAction.toAlgHom` is bijective. -/
theorem toAlgHom_bijective [Finite G] [FaithfulSMul G F] :
    Function.Bijective (MulSemiringAction.toAlgHom _ _ : G → F →ₐ[subfield G F] F) := by
  /-
    G : Type u_1
    F : Type u_2
    inst✝⁴ : Group G
    inst✝³ : Field F
    inst✝² : MulSemiringAction G F
    inst✝¹ : Finite G
    inst✝ : FaithfulSMul G F
    ⊢ Function.Bijective (MulSemiringAction.toAlgHom (Subtype fun x => Membership. …
  -/
  cases nonempty_fintype G
  /-
    case intro
    G : Type u_1
    F : Type u_2
    inst✝⁴ : Group G
    inst✝³ : Field F
    inst✝² : MulSemiringAction G F
    inst✝¹ : Finite G
    inst✝ : FaithfulSMul G F
    val✝ : Fintype G
    ⊢ Function.Bijective (MulSemiringAction.toAlgHom (Subtype fun x => Membership. …
  -/
  rw [Fintype.bijective_iff_injective_and_card]
  /-
    case intro
    G : Type u_1
    F : Type u_2
    inst✝⁴ : Group G
    inst✝³ : Field F
    inst✝² : MulSemiringAction G F
    inst✝¹ : Finite G
    inst✝ : FaithfulSMul G F
    val✝ : Fintype G
    ⊢ And (Function.Injective (MulSemiringAction.toAlgHom (Subtype fun x => Member …
  -/
  constructor
    /-
      case intro.left
      G : Type u_1
      F : Type u_2
      inst✝⁴ : Group G
      inst✝³ : Field F
      inst✝² : MulSemiringAction G F
      inst✝¹ : Finite G
      inst✝ : FaithfulSMul G F
      val✝ : Fintype G
      ⊢ Function.Injective (MulSemiringAction.toAlgHom (Subtype fun x => Membership. …
    -/
  · exact MulSemiringAction.toAlgHom_injective _ F
    /-
      🎉 no goals
    -/
    /-
      case intro.right
      G : Type u_1
      F : Type u_2
      inst✝⁴ : Group G
      inst✝³ : Field F
      inst✝² : MulSemiringAction G F
      inst✝¹ : Finite G
      inst✝ : FaithfulSMul G F
      val✝ : Fintype G
      ⊢ Eq (Fintype.card G) (Fintype.card (AlgHom (Subtype fun x => Membership.mem ( …
    -/
  · apply le_antisymm
      /-
        case intro.right.a
        G : Type u_1
        F : Type u_2
        inst✝⁴ : Group G
        inst✝³ : Field F
        inst✝² : MulSemiringAction G F
        inst✝¹ : Finite G
        inst✝ : FaithfulSMul G F
        val✝ : Fintype G
        ⊢ LE.le (Fintype.card G) (Fintype.card (AlgHom (Subtype fun x => Membership.me …
      -/
    · exact Fintype.card_le_of_injective _ (MulSemiringAction.toAlgHom_injective _ F)
      /-
        🎉 no goals
      -/
      /-
        case intro.right.a
        G : Type u_1
        F : Type u_2
        inst✝⁴ : Group G
        inst✝³ : Field F
        inst✝² : MulSemiringAction G F
        inst✝¹ : Finite G
        inst✝ : FaithfulSMul G F
        val✝ : Fintype G
        ⊢ LE.le (Fintype.card (AlgHom (Subtype fun x => Membership.mem (FixedPoints.su …
      -/
    · rw [← finrank_eq_card G F]
      /-
        case intro.right.a
        G : Type u_1
        F : Type u_2
        inst✝⁴ : Group G
        inst✝³ : Field F
        inst✝² : MulSemiringAction G F
        inst✝¹ : Finite G
        inst✝ : FaithfulSMul G F
        val✝ : Fintype G
        ⊢ LE.le (Fintype.card (AlgHom (Subtype fun x => Membership.mem (FixedPoints.su …
      -/
      exact LE.le.trans_eq (finrank_algHom _ F) (finrank_linearMap_self _ _ _)
      /-
        🎉 no goals
      -/


/-- Bijection between `G` and algebra endomorphisms of `F` that fix the fixed points. -/
def toAlgHomEquiv [Finite G] [FaithfulSMul G F] : G ≃ (F →ₐ[FixedPoints.subfield G F] F) :=
  Equiv.ofBijective _ (toAlgHom_bijective G F)


/-- `MulSemiringAction.toAlgAut` is bijective. -/
theorem toAlgAut_bijective [Finite G] [FaithfulSMul G F] :
    Function.Bijective (MulSemiringAction.toAlgAut G (FixedPoints.subfield G F) F) := by
  refine ⟨fun _ _ h ↦ (FixedPoints.toAlgHom_bijective G F).injective ?_,
    fun f ↦ ((FixedPoints.toAlgHom_bijective G F).surjective f).imp (fun _ h ↦ ?_)⟩ <;>
      /-
        case refine_1
        G : Type u_1
        F : Type u_2
        inst✝⁴ : Group G
        inst✝³ : Field F
        inst✝² : MulSemiringAction G F
        inst✝¹ : Finite G
        inst✝ : FaithfulSMul G F
        x✝¹ x✝ : G
        h : Eq ((MulSemiringAction.toAlgAut G (Subtype fun x => Membership.mem (FixedP …
        ⊢ Eq (MulSemiringAction.toAlgHom (Subtype fun x => Membership.mem (FixedPoints …
      -/
      /-
        🎉 no goals
      -/
      rwa [DFunLike.ext_iff] at h ⊢
      /-
        🎉 no goals
      -/


/-- Bijection between `G` and algebra automorphisms of `F` that fix the fixed points. -/
def toAlgAutMulEquiv [Finite G] [FaithfulSMul G F] : G ≃* (F ≃ₐ[FixedPoints.subfield G F] F) :=
  MulEquiv.ofBijective _ (toAlgAut_bijective G F)


/-- `MulSemiringAction.toAlgAut` is surjective. -/
theorem toAlgAut_surjective [Finite G] :
    Function.Surjective (MulSemiringAction.toAlgAut G (FixedPoints.subfield G F) F) := by
  let f : G →* F ≃ₐ[FixedPoints.subfield G F] F :=
    MulSemiringAction.toAlgAut G (FixedPoints.subfield G F) F
  /-
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfie …
    ⊢ Function.Surjective ⇑(MulSemiringAction.toAlgAut G (Subtype fun x => Members …
  -/
  let Q := G ⧸ f.ker
  /-
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfie …
    Q : Type u_1 := HasQuotient.Quotient G f.ker
    ⊢ Function.Surjective ⇑(MulSemiringAction.toAlgAut G (Subtype fun x => Members …
  -/
  let _ : MulSemiringAction Q F := MulSemiringAction.compHom _ (QuotientGroup.kerLift f)
  have : FaithfulSMul Q F := ⟨by
    intro q₁ q₂
    refine Quotient.inductionOn₂' q₁ q₂ (fun g₁ g₂ h ↦ QuotientGroup.eq.mpr ?_)
    rwa [MonoidHom.mem_ker, map_mul, map_inv, inv_mul_eq_one, AlgEquiv.ext_iff]⟩
  /-
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfie …
    Q : Type u_1 := HasQuotient.Quotient G f.ker
    x✝ : MulSemiringAction Q F := MulSemiringAction.compHom F (QuotientGroup.kerLi …
    this : FaithfulSMul Q F
    ⊢ Function.Surjective ⇑(MulSemiringAction.toAlgAut G (Subtype fun x => Members …
  -/
  intro f
  obtain ⟨q, hq⟩ := (toAlgAut_bijective Q F).surjective
    (AlgEquiv.ofRingEquiv (f := f) (fun ⟨x, hx⟩ ↦ f.commutes' ⟨x, fun g ↦ hx g⟩))
  /-
    case intro
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f✝ : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfi …
    Q : Type u_1 := HasQuotient.Quotient G f✝.ker
    x✝ : MulSemiringAction Q F := MulSemiringAction.compHom F (QuotientGroup.kerLi …
    this : FaithfulSMul Q F
    f : AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x) F F
    q : Q
    hq : Eq ((MulSemiringAction.toAlgAut Q (Subtype fun x => Membership.mem (Fixed …
    ⊢ Exists fun a => Eq ((MulSemiringAction.toAlgAut G (Subtype fun x => Membersh …
  -/
  revert hq
  /-
    case intro
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f✝ : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfi …
    Q : Type u_1 := HasQuotient.Quotient G f✝.ker
    x✝ : MulSemiringAction Q F := MulSemiringAction.compHom F (QuotientGroup.kerLi …
    this : FaithfulSMul Q F
    f : AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x) F F
    q : Q
    ⊢ Eq ((MulSemiringAction.toAlgAut Q (Subtype fun x => Membership.mem (FixedPoi …
  -/
  refine QuotientGroup.induction_on q (fun g hg ↦ ⟨g, ?_⟩)
  /-
    case intro
    G : Type u_1
    F : Type u_2
    inst✝³ : Group G
    inst✝² : Field F
    inst✝¹ : MulSemiringAction G F
    inst✝ : Finite G
    f✝ : MonoidHom G (AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfi …
    Q : Type u_1 := HasQuotient.Quotient G f✝.ker
    x✝ : MulSemiringAction Q F := MulSemiringAction.compHom F (QuotientGroup.kerLi …
    this : FaithfulSMul Q F
    f : AlgEquiv (Subtype fun x => Membership.mem (FixedPoints.subfield G F) x) F F
    q : Q
    g : G
    hg : Eq ((MulSemiringAction.toAlgAut Q (Subtype fun x => Membership.mem (Fixed …
    ⊢ Eq ((MulSemiringAction.toAlgAut G (Subtype fun x => Membership.mem (FixedPoi …
  -/
  rwa [AlgEquiv.ext_iff] at hg ⊢
  /-
    🎉 no goals
  -/



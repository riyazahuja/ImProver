open Classical in
/-- Non-computably choose an irreducible factor from a polynomial. -/
def factor (f : K[X]) : K[X] :=
  if H : ∃ g, Irreducible g ∧ g ∣ f then Classical.choose H else X


theorem irreducible_factor (f : K[X]) : Irreducible (factor f) := by
  /-
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    ⊢ Irreducible f.factor
  -/
  rw [factor]
  /-
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    ⊢ Irreducible (dite (Exists fun g => And (Irreducible g) (Dvd.dvd g f)) (fun H …
  -/
  split_ifs with H
    /-
      case pos
      K : Type v
      inst✝ : Field K
      f : Polynomial K
      H : Exists fun g => And (Irreducible g) (Dvd.dvd g f)
      ⊢ Irreducible (Classical.choose H)
    -/
  · exact (Classical.choose_spec H).1
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type v
      inst✝ : Field K
      f : Polynomial K
      H : Not (Exists fun g => And (Irreducible g) (Dvd.dvd g f))
      ⊢ Irreducible Polynomial.X
    -/
  · exact irreducible_X
    /-
      🎉 no goals
    -/


/-- See note [fact non-instances]. -/
theorem fact_irreducible_factor (f : K[X]) : Fact (Irreducible (factor f)) :=
  ⟨irreducible_factor f⟩


theorem factor_dvd_of_not_isUnit {f : K[X]} (hf1 : ¬IsUnit f) : factor f ∣ f := by
  /-
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    hf1 : Not (IsUnit f)
    ⊢ Dvd.dvd f.factor f
  -/
  by_cases hf2 : f = 0; · rw [hf2]; exact dvd_zero _
                                    /-
                                      🎉 no goals
                                    -/
  /-
    case neg
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    hf1 : Not (IsUnit f)
    hf2 : Not (Eq f 0)
    ⊢ Dvd.dvd f.factor f
  -/
  rw [factor, dif_pos (WfDvdMonoid.exists_irreducible_factor hf1 hf2)]
  /-
    case neg
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    hf1 : Not (IsUnit f)
    hf2 : Not (Eq f 0)
    ⊢ Dvd.dvd (Classical.choose ⋯) f
  -/
  exact (Classical.choose_spec <| WfDvdMonoid.exists_irreducible_factor hf1 hf2).2
  /-
    🎉 no goals
  -/


theorem factor_dvd_of_degree_ne_zero {f : K[X]} (hf : f.degree ≠ 0) : factor f ∣ f :=
  factor_dvd_of_not_isUnit (mt degree_eq_zero_of_isUnit hf)


theorem factor_dvd_of_natDegree_ne_zero {f : K[X]} (hf : f.natDegree ≠ 0) : factor f ∣ f :=
  factor_dvd_of_degree_ne_zero (mt natDegree_eq_of_degree_eq_some hf)


lemma isCoprime_iff_aeval_ne_zero (f g : K[X]) : IsCoprime f g ↔ ∀ {A : Type v} [CommRing A]
    [IsDomain A] [Algebra K A] (a : A), aeval a f ≠ 0 ∨ aeval a g ≠ 0 := by
  /-
    K : Type v
    inst✝ : Field K
    f g : Polynomial K
    ⊢ Iff (IsCoprime f g) (∀ {A : Type v} [inst : CommRing A] [inst_1 : IsDomain A …
  -/
  refine ⟨fun h => aeval_ne_zero_of_isCoprime h, fun h => isCoprime_of_dvd _ _ ?_ fun x hx _ => ?_⟩
    /-
      case refine_1
      K : Type v
      inst✝ : Field K
      f g : Polynomial K
      h : ∀ {A : Type v} [inst : CommRing A] [inst_1 : IsDomain A] [inst_2 : Algebra …
      ⊢ Not (And (Eq f 0) (Eq g 0))
    -/
  · replace h := @h K _ _ _ 0
    /-
      case refine_1
      K : Type v
      inst✝ : Field K
      f g : Polynomial K
      h : Or (Ne ((Polynomial.aeval 0) f) 0) (Ne ((Polynomial.aeval 0) g) 0)
      ⊢ Not (And (Eq f 0) (Eq g 0))
    -/
    contrapose! h
    /-
      case refine_1
      K : Type v
      inst✝ : Field K
      f g : Polynomial K
      h : And (Eq f 0) (Eq g 0)
      ⊢ And (Eq ((Polynomial.aeval 0) f) 0) (Eq ((Polynomial.aeval 0) g) 0)
    -/
    rw [h.left, h.right, map_zero, and_self]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type v
      inst✝ : Field K
      f g : Polynomial K
      h : ∀ {A : Type v} [inst : CommRing A] [inst_1 : IsDomain A] [inst_2 : Algebra …
      x : Polynomial K
      hx : Membership.mem (nonunits (Polynomial K)) x
      x✝ : Ne x 0
      ⊢ Dvd.dvd x f → Not (Dvd.dvd x g)
    -/
  · rintro ⟨_, rfl⟩ ⟨_, rfl⟩
    /-
      case refine_2.intro.intro
      K : Type v
      inst✝ : Field K
      x : Polynomial K
      hx : Membership.mem (nonunits (Polynomial K)) x
      x✝ : Ne x 0
      w✝¹ w✝ : Polynomial K
      h : ∀ {A : Type v} [inst : CommRing A] [inst_1 : IsDomain A] [inst_2 : Algebra …
      ⊢ False
    -/
    replace h := not_and_or.mpr <| h <| AdjoinRoot.root x.factor
    simp only [AdjoinRoot.aeval_eq, AdjoinRoot.mk_eq_zero,
      dvd_mul_of_dvd_left <| factor_dvd_of_not_isUnit hx, true_and, not_true] at h


/-- Divide a polynomial f by `X - C r` where `r` is a root of `f` in a bigger field extension. -/
def removeFactor (f : K[X]) : Polynomial (AdjoinRoot <| factor f) :=
  map (AdjoinRoot.of f.factor) f /ₘ (X - C (AdjoinRoot.root f.factor))


theorem X_sub_C_mul_removeFactor (f : K[X]) (hf : f.natDegree ≠ 0) :
    (X - C (AdjoinRoot.root f.factor)) * f.removeFactor = map (AdjoinRoot.of f.factor) f := by
  /-
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    hf : Ne f.natDegree 0
    ⊢ Eq (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C (AdjoinRoot.root f.facto …
  -/
  let ⟨g, hg⟩ := factor_dvd_of_natDegree_ne_zero hf
  apply (mul_divByMonic_eq_iff_isRoot
    (R := AdjoinRoot f.factor) (a := AdjoinRoot.root f.factor)).mpr
  /-
    K : Type v
    inst✝ : Field K
    f : Polynomial K
    hf : Ne f.natDegree 0
    g : Polynomial K
    hg : Eq f (HMul.hMul f.factor g)
    ⊢ (Polynomial.map (AdjoinRoot.of f.factor) f).IsRoot (AdjoinRoot.root f.factor)
  -/
  rw [IsRoot.def, eval_map, hg, eval₂_mul, ← hg, AdjoinRoot.eval₂_root, zero_mul]
  /-
    🎉 no goals
  -/


theorem natDegree_removeFactor (f : K[X]) : f.removeFactor.natDegree = f.natDegree - 1 := by
  -- Porting note: `(map (AdjoinRoot.of f.factor) f)` was `_`
  rw [removeFactor, natDegree_divByMonic (map (AdjoinRoot.of f.factor) f) (monic_X_sub_C _),
    natDegree_map, natDegree_X_sub_C]


theorem natDegree_removeFactor' {f : K[X]} {n : ℕ} (hfn : f.natDegree = n + 1) :
                                       /-
                                         K : Type v
                                         inst✝ : Field K
                                         f : Polynomial K
                                         n : Nat
                                         hfn : Eq f.natDegree (HAdd.hAdd n 1)
                                         ⊢ Eq f.removeFactor.natDegree n
                                       -/
    f.removeFactor.natDegree = n := by rw [natDegree_removeFactor, hfn, n.add_sub_cancel]
                                       /-
                                         🎉 no goals
                                       -/


/-- Auxiliary construction to a splitting field of a polynomial, which removes
`n` (arbitrarily-chosen) factors.

It constructs the type, proves that is a field and algebra over the base field.

Uses recursion on the degree.
-/
def SplittingFieldAuxAux (n : ℕ) : ∀ {K : Type u} [Field K], K[X] →
    Σ (L : Type u) (_ : Field L), Algebra K L :=
  -- Porting note: added motive
  Nat.recOn (motive := fun (_x : ℕ) => ∀ {K : Type u} [_inst_4 : Field K], K[X] →
      Σ (L : Type u) (_ : Field L), Algebra K L) n
    (fun {K} _ _ => ⟨K, inferInstance, inferInstance⟩)
    fun _ ih _ _ f =>
      let ⟨L, fL, _⟩ := ih f.removeFactor
      ⟨L, fL, (RingHom.comp (algebraMap _ _) (AdjoinRoot.of f.factor)).toAlgebra⟩


/-- Auxiliary construction to a splitting field of a polynomial, which removes
`n` (arbitrarily-chosen) factors. It is the type constructed in `SplittingFieldAuxAux`.
-/
def SplittingFieldAux (n : ℕ) {K : Type u} [Field K] (f : K[X]) : Type u :=
  (SplittingFieldAuxAux n f).1


instance SplittingFieldAux.field (n : ℕ) {K : Type u} [Field K] (f : K[X]) :
    Field (SplittingFieldAux n f) :=
  (SplittingFieldAuxAux n f).2.1


instance (n : ℕ) {K : Type u} [Field K] (f : K[X]) : Inhabited (SplittingFieldAux n f) :=
  ⟨0⟩


instance SplittingFieldAux.algebra (n : ℕ) {K : Type u} [Field K] (f : K[X]) :
    Algebra K (SplittingFieldAux n f) :=
  (SplittingFieldAuxAux n f).2.2


theorem succ (n : ℕ) (f : K[X]) :
    SplittingFieldAux (n + 1) f = SplittingFieldAux n f.removeFactor :=
  rfl


instance algebra''' {n : ℕ} {f : K[X]} :
    Algebra (AdjoinRoot f.factor) (SplittingFieldAux n f.removeFactor) :=
  SplittingFieldAux.algebra n _


instance algebra' {n : ℕ} {f : K[X]} : Algebra (AdjoinRoot f.factor) (SplittingFieldAux n.succ f) :=
  SplittingFieldAux.algebra'''


instance algebra'' {n : ℕ} {f : K[X]} : Algebra K (SplittingFieldAux n f.removeFactor) :=
  RingHom.toAlgebra (RingHom.comp (algebraMap _ _) (AdjoinRoot.of f.factor))


instance scalar_tower' {n : ℕ} {f : K[X]} :
    IsScalarTower K (AdjoinRoot f.factor) (SplittingFieldAux n f.removeFactor) :=
  IsScalarTower.of_algebraMap_eq fun _ => rfl


theorem algebraMap_succ (n : ℕ) (f : K[X]) :
    algebraMap K (SplittingFieldAux (n + 1) f) =
      (algebraMap (AdjoinRoot f.factor) (SplittingFieldAux n f.removeFactor)).comp
        (AdjoinRoot.of f.factor) :=
  rfl


protected theorem splits (n : ℕ) :
    ∀ {K : Type u} [Field K],
      ∀ (f : K[X]) (_hfn : f.natDegree = n), Splits (algebraMap K <| SplittingFieldAux n f) f :=
  Nat.recOn (motive := fun n => ∀ {K : Type u} [Field K],
      ∀ (f : K[X]) (_hfn : f.natDegree = n), Splits (algebraMap K <| SplittingFieldAux n f) f) n
    (fun {_} _ _ hf =>
      splits_of_degree_le_one _
        (le_trans degree_le_natDegree <| hf.symm ▸ WithBot.coe_le_coe.2 zero_le_one))
    fun n ih {K} _ f hf => by
    rw [← splits_id_iff_splits, algebraMap_succ, ← map_map, splits_id_iff_splits,
      ← X_sub_C_mul_removeFactor f fun h => by rw [h] at hf; cases hf]
    /-
      n✝ : Nat
      K✝ : Type u
      inst✝ : Field K✝
      n : Nat
      ih : (fun n => ∀ {K : Type u} [inst : Field K] (f : Polynomial K), Eq f.natDeg …
      K : Type u
      x✝ : Field K
      f : Polynomial K
      hf : Eq f.natDegree n.succ
      ⊢ Polynomial.Splits (algebraMap (AdjoinRoot f.factor) (Polynomial.SplittingFie …
    -/
    exact splits_mul _ (splits_X_sub_C _) (ih _ (natDegree_removeFactor' hf))
    /-
      🎉 no goals
    -/


theorem adjoin_rootSet (n : ℕ) :
    ∀ {K : Type u} [Field K],
      ∀ (f : K[X]) (_hfn : f.natDegree = n),
        Algebra.adjoin K (f.rootSet (SplittingFieldAux n f)) = ⊤ :=
  Nat.recOn (motive := fun n =>
    ∀ {K : Type u} [Field K],
      ∀ (f : K[X]) (_hfn : f.natDegree = n),
        Algebra.adjoin K (f.rootSet (SplittingFieldAux n f)) = ⊤)
    n (fun {_} _ _ _hf => Algebra.eq_top_iff.2 fun x => Subalgebra.range_le _ ⟨x, rfl⟩)
    fun n ih {K} _ f hfn => by
    /-
      n✝ : Nat
      K✝ : Type u
      inst✝ : Field K✝
      n : Nat
      ih : (fun n => ∀ {K : Type u} [inst : Field K] (f : Polynomial K), Eq f.natDeg …
      K : Type u
      x✝ : Field K
      f : Polynomial K
      hfn : Eq f.natDegree n.succ
      ⊢ Eq (Algebra.adjoin K (f.rootSet (Polynomial.SplittingFieldAux n.succ f))) To …
    -/
    have hndf : f.natDegree ≠ 0 := by intro h; rw [h] at hfn; cases hfn
    /-
      n✝ : Nat
      K✝ : Type u
      inst✝ : Field K✝
      n : Nat
      ih : (fun n => ∀ {K : Type u} [inst : Field K] (f : Polynomial K), Eq f.natDeg …
      K : Type u
      x✝ : Field K
      f : Polynomial K
      hfn : Eq f.natDegree n.succ
      hndf : Ne f.natDegree 0
      ⊢ Eq (Algebra.adjoin K (f.rootSet (Polynomial.SplittingFieldAux n.succ f))) To …
    -/
    have hfn0 : f ≠ 0 := by intro h; rw [h] at hndf; exact hndf rfl
    /-
      n✝ : Nat
      K✝ : Type u
      inst✝ : Field K✝
      n : Nat
      ih : (fun n => ∀ {K : Type u} [inst : Field K] (f : Polynomial K), Eq f.natDeg …
      K : Type u
      x✝ : Field K
      f : Polynomial K
      hfn : Eq f.natDegree n.succ
      hndf : Ne f.natDegree 0
      hfn0 : Ne f 0
      ⊢ Eq (Algebra.adjoin K (f.rootSet (Polynomial.SplittingFieldAux n.succ f))) To …
    -/
    have hmf0 : map (algebraMap K (SplittingFieldAux n.succ f)) f ≠ 0 := map_ne_zero hfn0
    classical
    rw [rootSet_def, aroots_def]
    rw [algebraMap_succ, ← map_map, ← X_sub_C_mul_removeFactor _ hndf, Polynomial.map_mul] at hmf0 ⊢
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    erw [roots_mul hmf0, Polynomial.map_sub, map_X, map_C, roots_X_sub_C, Multiset.toFinset_add,
      Finset.coe_union, Multiset.toFinset_singleton, Finset.coe_singleton,
      Algebra.adjoin_union_eq_adjoin_adjoin, ← Set.image_singleton,
      Algebra.adjoin_algebraMap K (SplittingFieldAux n f.removeFactor),
      AdjoinRoot.adjoinRoot_eq_top, Algebra.map_top]
    /- Porting note: was `rw [IsScalarTower.adjoin_range_toAlgHom K (AdjoinRoot f.factor)
        (SplittingFieldAux n f.removeFactor)]` -/
    have := IsScalarTower.adjoin_range_toAlgHom K (AdjoinRoot f.factor)
        (SplittingFieldAux n f.removeFactor)
        (f.removeFactor.rootSet (SplittingFieldAux n f.removeFactor))
    refine this.trans ?_
    rw [ih _ (natDegree_removeFactor' hfn), Subalgebra.restrictScalars_top]


instance (f : K[X]) : IsSplittingField K (SplittingFieldAux f.natDegree f) f :=
  ⟨SplittingFieldAux.splits _ _ rfl, SplittingFieldAux.adjoin_rootSet _ _ rfl⟩


/-- A splitting field of a polynomial. -/
@[stacks 09HV "The construction of the splitting field."]
def SplittingField (f : K[X]) :=
  MvPolynomial (SplittingFieldAux f.natDegree f) K ⧸
    RingHom.ker (MvPolynomial.aeval (R := K) id).toRingHom


instance commRing : CommRing (SplittingField f) :=
  Ideal.Quotient.commRing _


instance inhabited : Inhabited (SplittingField f) :=
  ⟨37⟩


instance {S : Type*} [DistribSMul S K] [IsScalarTower S K K] : SMul S (SplittingField f) :=
  Submodule.Quotient.instSMul' _


instance algebra : Algebra K (SplittingField f) :=
  Ideal.Quotient.algebra _


instance algebra' {R : Type*} [CommSemiring R] [Algebra R K] : Algebra R (SplittingField f) :=
  Ideal.Quotient.algebra _


instance isScalarTower {R : Type*} [CommSemiring R] [Algebra R K] :
    IsScalarTower R K (SplittingField f) :=
  Ideal.Quotient.isScalarTower _ _ _


/-- The algebra equivalence with `SplittingFieldAux`,
which we will use to construct the field structure. -/
def algEquivSplittingFieldAux (f : K[X]) : SplittingField f ≃ₐ[K] SplittingFieldAux f.natDegree f :=
                                                                       /-
                                                                         F : Type u
                                                                         K : Type v
                                                                         L : Type w
                                                                         inst✝² : Field K
                                                                         inst✝¹ : Field L
                                                                         inst✝ : Field F
                                                                         f✝ f : Polynomial K
                                                                         x : Polynomial.SplittingFieldAux f.natDegree f
                                                                         ⊢ Eq ((MvPolynomial.aeval id) (MvPolynomial.X x)) x
                                                                       -/
  Ideal.quotientKerAlgEquivOfSurjective fun x => ⟨MvPolynomial.X x, by simp⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance instGroupWithZero : GroupWithZero (SplittingField f) :=
  let e := algEquivSplittingFieldAux f
  { inv := fun a ↦ e.symm (e a)⁻¹
                   /-
                     F : Type u
                     K : Type v
                     L : Type w
                     inst✝² : Field K
                     inst✝¹ : Field L
                     inst✝ : Field F
                     f : Polynomial K
                     e : AlgEquiv K f.SplittingField (Polynomial.SplittingFieldAux f.natDegree f) : …
                     ⊢ Eq (Inv.inv 0) 0
                   -/
    inv_zero := by simp
                   /-
                     🎉 no goals
                   -/
                                                   /-
                                                     F : Type u
                                                     K : Type v
                                                     L : Type w
                                                     inst✝² : Field K
                                                     inst✝¹ : Field L
                                                     inst✝ : Field F
                                                     f : Polynomial K
                                                     e : AlgEquiv K f.SplittingField (Polynomial.SplittingFieldAux f.natDegree f) : …
                                                     a : f.SplittingField
                                                     ha : Ne a 0
                                                     ⊢ Eq (e (HMul.hMul a (Inv.inv a))) (e 1)
                                                   -/
    mul_inv_cancel := fun a ha ↦ e.injective <| by simp [EmbeddingLike.map_ne_zero_iff.2 ha]
                                                   /-
                                                     🎉 no goals
                                                   -/
    __ := e.surjective.nontrivial }


instance instField : Field (SplittingField f) where
  __ := commRing _
  __ := instGroupWithZero _
  nnratCast q := algebraMap K _ q
  ratCast q := algebraMap K _ q
  nnqsmul := (· • ·)
  qsmul := (· • ·)
                        /-
                          F : Type u
                          K : Type v
                          L : Type w
                          inst✝² : Field K
                          inst✝¹ : Field L
                          inst✝ : Field F
                          f : Polynomial K
                          q : NNRat
                          ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
                        -/
  nnratCast_def q := by change algebraMap K _ _ = _; simp_rw [NNRat.cast_def, map_div₀, map_natCast]
                                                     /-
                                                       🎉 no goals
                                                     -/
  ratCast_def q := by
    /-
      F : Type u
      K : Type v
      L : Type w
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Field F
      f : Polynomial K
      q : Rat
      ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
    -/
    change algebraMap K _ _ = _; rw [Rat.cast_def, map_div₀, map_intCast, map_natCast]
    /-
      F : Type u
      K : Type v
      L : Type w
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Field F
      f : Polynomial K
      q : NNRat
      x : f.SplittingField
      p : MvPolynomial (Polynomial.SplittingFieldAux f.natDegree f) K
      ⊢ Eq ((fun x => HSMul.hSMul q x) p) ((fun x1 x2 => HMul.hMul x1 x2) ((algebraM …
    -/
                                 /-
                                   🎉 no goals
                                 -/
         /-
           🎉 no goals
         -/
  nnqsmul_def q x := Quotient.inductionOn x fun p ↦ congr_arg Quotient.mk'' <| by
    ext; simp [MvPolynomial.algebraMap_eq, NNRat.smul_def]
  qsmul_def q x := Quotient.inductionOn x fun p ↦ congr_arg Quotient.mk'' <| by
    /-
      F : Type u
      K : Type v
      L : Type w
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Field F
      f : Polynomial K
      q : Rat
      x : f.SplittingField
      p : MvPolynomial (Polynomial.SplittingFieldAux f.natDegree f) K
      ⊢ Eq ((fun x => HSMul.hSMul q x) p) ((fun x1 x2 => HMul.hMul x1 x2) ((algebraM …
    -/
    ext; simp [MvPolynomial.algebraMap_eq, Rat.smul_def]
         /-
           🎉 no goals
         -/


instance instCharZero [CharZero K] : CharZero (SplittingField f) :=
  charZero_of_injective_algebraMap (algebraMap K _).injective


instance instCharP (p : ℕ) [CharP K p] : CharP (SplittingField f) p :=
  charP_of_injective_algebraMap (algebraMap K _).injective p


instance instExpChar (p : ℕ) [ExpChar K p] : ExpChar (SplittingField f) p :=
  expChar_of_injective_algebraMap (algebraMap K _).injective p


instance _root_.Polynomial.IsSplittingField.splittingField (f : K[X]) :
    IsSplittingField K (SplittingField f) f :=
  IsSplittingField.of_algEquiv _ f (algEquivSplittingFieldAux f).symm


@[stacks 09HU "Splitting part"]
protected theorem splits : Splits (algebraMap K (SplittingField f)) f :=
  IsSplittingField.splits f.SplittingField f


/-- Embeds the splitting field into any other field that splits the polynomial. -/
def lift : SplittingField f →ₐ[K] L :=
  IsSplittingField.lift f.SplittingField f hb


theorem adjoin_rootSet : Algebra.adjoin K (f.rootSet (SplittingField f)) = ⊤ :=
  Polynomial.IsSplittingField.adjoin_rootSet _ f


instance (f : K[X]) : FiniteDimensional K f.SplittingField :=
  finiteDimensional f.SplittingField f


instance [Finite K] (f : K[X]) : Finite f.SplittingField :=
  Module.finite_of_finite K


instance (f : K[X]) : NoZeroSMulDivisors K f.SplittingField :=
  inferInstance


/-- Any splitting field is isomorphic to `SplittingFieldAux f`. -/
def algEquiv (f : K[X]) [h : IsSplittingField K L f] : L ≃ₐ[K] SplittingField f :=
  AlgEquiv.ofBijective (lift L f <| splits (SplittingField f) f) <|
    have := finiteDimensional L f
    ((Algebra.IsAlgebraic.of_finite K L).algHom_bijective₂ _ <| lift _ f h.1).1



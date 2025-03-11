local notation "φ" => LieHom.toLinearMap (LieModule.toEnd R L M)


/--
Let `M` be a representation of a Lie algebra `L` over a nontrivial commutative ring `R`,
and assume that `L` and `M` are finite free as `R`-module.
Then the coefficients of the characteristic polynomial of `⁅x, ·⁆` are polynomial in `x`.
The *rank* of `M` is the smallest `n` for which the `n`-th coefficient is not the zero polynomial.
-/
noncomputable
def rank : ℕ := nilRank φ


lemma polyCharpoly_coeff_rank_ne_zero [Nontrivial R] [DecidableEq ι] :
    (polyCharpoly φ b).coeff (rank R L M) ≠ 0 :=
  polyCharpoly_coeff_nilRank_ne_zero _ _


lemma rank_eq_natTrailingDegree [Nontrivial R] [DecidableEq ι] :
    rank R L M = (polyCharpoly φ b).natTrailingDegree := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    ι : Type u_5
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    inst✝¹⁰ : Module.Finite R L
    inst✝⁹ : Module.Free R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : Module.Finite R M
    inst✝³ : Module.Free R M
    inst✝² : Fintype ι
    b : Basis ι R L
    inst✝¹ : Nontrivial R
    inst✝ : DecidableEq ι
    ⊢ Eq (LieModule.rank R L M) ((↑(LieModule.toEnd R L M)).polyCharpoly b).natTra …
  -/
  apply nilRank_eq_polyCharpoly_natTrailingDegree
  /-
    🎉 no goals
  -/


include bₘ in
lemma rank_le_card [Nontrivial R] : rank R L M ≤ Fintype.card ιₘ :=
  nilRank_le_card _ bₘ


lemma rank_le_finrank [Nontrivial R] : rank R L M ≤ finrank R M :=
  nilRank_le_finrank _


lemma rank_le_natTrailingDegree_charpoly_ad [Nontrivial R] :
    rank R L M ≤ (toEnd R L M x).charpoly.natTrailingDegree :=
  nilRank_le_natTrailingDegree_charpoly _ _


/-- Let `x` be an element of a Lie algebra `L` over `R`, and write `n` for `rank R L`.
Then `x` is *regular*
if the `n`-th coefficient of the characteristic polynomial of `ad R L x` is non-zero. -/
def IsRegular (x : L) : Prop := LinearMap.IsNilRegular φ x


lemma isRegular_def :
    IsRegular R M x ↔ (toEnd R L M x).charpoly.coeff (rank R L M) ≠ 0 := Iff.rfl


lemma isRegular_iff_coeff_polyCharpoly_rank_ne_zero [DecidableEq ι] :
    IsRegular R M x ↔
    MvPolynomial.eval (b.repr x)
      ((polyCharpoly φ b).coeff (rank R L M)) ≠ 0 :=
  LinearMap.isNilRegular_iff_coeff_polyCharpoly_nilRank_ne_zero _ _ _


lemma isRegular_iff_natTrailingDegree_charpoly_eq_rank [Nontrivial R] :
    IsRegular R M x ↔ (toEnd R L M x).charpoly.natTrailingDegree = rank R L M :=
  LinearMap.isNilRegular_iff_natTrailingDegree_charpoly_eq_nilRank _ _

open Cardinal Module MvPolynomial in
lemma exists_isRegular_of_finrank_le_card (h : finrank R M ≤ #R) :
    ∃ x : L, IsRegular R M x :=
  LinearMap.exists_isNilRegular_of_finrank_le_card _ h


lemma exists_isRegular [Infinite R] : ∃ x : L, IsRegular R M x :=
  LinearMap.exists_isNilRegular _


/--
Let `L` be a Lie algebra over a nontrivial commutative ring `R`,
and assume that `L` is finite free as `R`-module.
Then the coefficients of the characteristic polynomial of `ad R L x` are polynomial in `x`.
The *rank* of `L` is the smallest `n` for which the `n`-th coefficient is not the zero polynomial.
-/
noncomputable
abbrev rank : ℕ := LieModule.rank R L L


lemma polyCharpoly_coeff_rank_ne_zero [Nontrivial R] [DecidableEq ι] :
    (polyCharpoly (ad R L).toLinearMap b).coeff (rank R L) ≠ 0 :=
  polyCharpoly_coeff_nilRank_ne_zero _ _


lemma rank_eq_natTrailingDegree [Nontrivial R] [DecidableEq ι] :
    rank R L = (polyCharpoly (ad R L).toLinearMap b).natTrailingDegree := by
  /-
    R : Type u_1
    L : Type u_3
    ι : Type u_5
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : Module.Finite R L
    inst✝³ : Module.Free R L
    inst✝² : Fintype ι
    b : Basis ι R L
    inst✝¹ : Nontrivial R
    inst✝ : DecidableEq ι
    ⊢ Eq (LieAlgebra.rank R L) ((↑(LieAlgebra.ad R L)).polyCharpoly b).natTrailing …
  -/
  apply nilRank_eq_polyCharpoly_natTrailingDegree
  /-
    🎉 no goals
  -/


include b in
lemma rank_le_card [Nontrivial R] : rank R L ≤ Fintype.card ι :=
  nilRank_le_card _ b


lemma rank_le_finrank [Nontrivial R] : rank R L ≤ finrank R L :=
  nilRank_le_finrank _


lemma rank_le_natTrailingDegree_charpoly_ad [Nontrivial R] :
    rank R L ≤ (ad R L x).charpoly.natTrailingDegree :=
  nilRank_le_natTrailingDegree_charpoly _ _


/-- Let `x` be an element of a Lie algebra `L` over `R`, and write `n` for `rank R L`.
Then `x` is *regular*
if the `n`-th coefficient of the characteristic polynomial of `ad R L x` is non-zero. -/
abbrev IsRegular (x : L) : Prop := LieModule.IsRegular R L x


lemma isRegular_def :
    IsRegular R x ↔ (Polynomial.coeff (ad R L x).charpoly (rank R L) ≠ 0) := Iff.rfl


lemma isRegular_iff_coeff_polyCharpoly_rank_ne_zero [DecidableEq ι] :
    IsRegular R x ↔
    MvPolynomial.eval (b.repr x)
      ((polyCharpoly (ad R L).toLinearMap b).coeff (rank R L)) ≠ 0 :=
  LinearMap.isNilRegular_iff_coeff_polyCharpoly_nilRank_ne_zero _ _ _


lemma isRegular_iff_natTrailingDegree_charpoly_eq_rank [Nontrivial R] :
    IsRegular R x ↔ (ad R L x).charpoly.natTrailingDegree = rank R L :=
  LinearMap.isNilRegular_iff_natTrailingDegree_charpoly_eq_nilRank _ _

open Cardinal Module MvPolynomial in
lemma exists_isRegular_of_finrank_le_card (h : finrank R L ≤ #R) :
    ∃ x : L, IsRegular R x :=
  LinearMap.exists_isNilRegular_of_finrank_le_card _ h


lemma exists_isRegular [Infinite R] : ∃ x : L, IsRegular R x :=
  LinearMap.exists_isNilRegular _


lemma finrank_engel (x : L) :
    finrank K (engel K x) = (ad K L x).charpoly.natTrailingDegree :=
  (ad K L x).finrank_maxGenEigenspace


lemma rank_le_finrank_engel (x : L) :
    rank K L ≤ finrank K (engel K x) :=
  (rank_le_natTrailingDegree_charpoly_ad K x).trans
    (finrank_engel K x).ge


lemma isRegular_iff_finrank_engel_eq_rank (x : L) :
    IsRegular K x ↔ finrank K (engel K x) = rank K L := by
  /-
    K : Type u_7
    L : Type u_8
    inst✝³ : Field K
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra K L
    inst✝ : Module.Finite K L
    x : L
    ⊢ Iff (LieAlgebra.IsRegular K x) (Eq (Module.finrank K (Subtype fun x_1 => Mem …
  -/
  rw [isRegular_iff_natTrailingDegree_charpoly_eq_rank, finrank_engel]
  /-
    🎉 no goals
  -/



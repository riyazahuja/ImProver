/-- A Sylow `p`-subgroup is a maximal `p`-subgroup. -/
structure Sylow extends Subgroup G where
  isPGroup' : IsPGroup p toSubgroup
  is_maximal' : ∀ {Q : Subgroup G}, IsPGroup p Q → toSubgroup ≤ Q → Q = toSubgroup


instance : CoeOut (Sylow p G) (Subgroup G) :=
  ⟨toSubgroup⟩

-- Porting note: syntactic tautology
-- @[simp]
-- theorem toSubgroup_eq_coe {P : Sylow p G} : P.toSubgroup = ↑P :=
--   rfl


@[ext]
                                                                       /-
                                                                         p : Nat
                                                                         G : Type u_1
                                                                         inst✝ : Group G
                                                                         P Q : Sylow p G
                                                                         h : Eq ↑P ↑Q
                                                                         ⊢ Eq P Q
                                                                       -/
theorem ext {P Q : Sylow p G} (h : (P : Subgroup G) = Q) : P = Q := by cases P; cases Q; congr
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance : SetLike (Sylow p G) G where
  coe := (↑)
  coe_injective' _ _ h := ext (SetLike.coe_injective h)


instance : SubgroupClass (Sylow p G) G where
  mul_mem := Subgroup.mul_mem _
  one_mem _ := Subgroup.one_mem _
  inv_mem := Subgroup.inv_mem _


/-- A `p`-subgroup with index indivisible by `p` is a Sylow subgroup. -/
def _root_.IsPGroup.toSylow [Fact p.Prime] {P : Subgroup G}
    (hP1 : IsPGroup p P) (hP2 : ¬ p ∣ P.index) : Sylow p G :=
  { P with
    isPGroup' := hP1
    is_maximal' := by
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : Fact (Nat.Prime p)
        P : Subgroup G
        hP1 : IsPGroup p (Subtype fun x => Membership.mem P x)
        hP2 : Not (Dvd.dvd p P.index)
        ⊢ ∀ {Q : Subgroup G}, IsPGroup p (Subtype fun x => Membership.mem Q x) → LE.le …
      -/
      intro Q hQ hPQ
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : Fact (Nat.Prime p)
        P : Subgroup G
        hP1 : IsPGroup p (Subtype fun x => Membership.mem P x)
        hP2 : Not (Dvd.dvd p P.index)
        Q : Subgroup G
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hPQ : LE.le P Q
        ⊢ Eq Q P
      -/
      have : P.FiniteIndex := ⟨fun h ↦ hP2 (h ▸ (dvd_zero p))⟩
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : Fact (Nat.Prime p)
        P : Subgroup G
        hP1 : IsPGroup p (Subtype fun x => Membership.mem P x)
        hP2 : Not (Dvd.dvd p P.index)
        Q : Subgroup G
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hPQ : LE.le P Q
        this : P.FiniteIndex
        ⊢ Eq Q P
      -/
      obtain ⟨k, hk⟩ := (hQ.to_quotient (P.normalCore.subgroupOf Q)).exists_card_eq
      /-
        case intro
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : Fact (Nat.Prime p)
        P : Subgroup G
        hP1 : IsPGroup p (Subtype fun x => Membership.mem P x)
        hP2 : Not (Dvd.dvd p P.index)
        Q : Subgroup G
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hPQ : LE.le P Q
        this : P.FiniteIndex
        k : Nat
        hk : Eq (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.mem Q x)  …
        ⊢ Eq Q P
      -/
      have h := hk ▸ Nat.Prime.coprime_pow_of_not_dvd (m := k) Fact.out hP2
      exact le_antisymm (Subgroup.relindex_eq_one.mp
        (Nat.eq_one_of_dvd_coprimes h (Subgroup.relindex_dvd_index_of_le hPQ)
        (Subgroup.relindex_dvd_of_le_left Q P.normalCore_le))) hPQ }


@[simp] theorem _root_.IsPGroup.toSylow_coe [Fact p.Prime] {P : Subgroup G}
    (hP1 : IsPGroup p P) (hP2 : ¬ p ∣ P.index) : (hP1.toSylow hP2) = P :=
  rfl


@[simp] theorem _root_.IsPGroup.mem_toSylow [Fact p.Prime] {P : Subgroup G}
    (hP1 : IsPGroup p P) (hP2 : ¬ p ∣ P.index) {g : G} : g ∈ hP1.toSylow hP2 ↔ g ∈ P :=
  .rfl


/-- A subgroup with cardinality `p ^ n` is a Sylow subgroup
 where `n` is the multiplicity of `p` in the group order. -/
def ofCard [Finite G] {p : ℕ} [Fact p.Prime] (H : Subgroup G)
    (card_eq : Nat.card H = p ^ (Nat.card G).factorization p) : Sylow p G :=
  (IsPGroup.of_card card_eq).toSylow (by
    /-
      p✝ : Nat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      H : Subgroup G
      card_eq : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p ((N …
      ⊢ Not (Dvd.dvd p H.index)
    -/
    rw [← mul_dvd_mul_iff_left (Nat.card_pos (α := H)).ne', card_mul_index, card_eq, ← pow_succ]
    /-
      p✝ : Nat
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      H : Subgroup G
      card_eq : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p ((N …
      ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd ((Nat.card G).factorization p) 1)) (Nat …
    -/
    exact Nat.pow_succ_factorization_not_dvd Nat.card_pos.ne' Fact.out)
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_ofCard [Finite G] {p : ℕ} [Fact p.Prime] (H : Subgroup G)
    (card_eq : Nat.card H = p ^ (Nat.card G).factorization p) : ofCard H card_eq = H :=
  rfl


/-- The preimage of a Sylow subgroup under a p-group-kernel homomorphism is a Sylow subgroup. -/
def comapOfKerIsPGroup (hϕ : IsPGroup p ϕ.ker) (h : P ≤ ϕ.range) : Sylow p K :=
  { P.1.comap ϕ with
    isPGroup' := P.2.comap_of_ker_isPGroup ϕ hϕ
    is_maximal' := fun {Q} hQ hle => by
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        P : Sylow p G
        K : Type u_2
        inst✝ : Group K
        ϕ : MonoidHom K G
        N : Subgroup G
        hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
        h : LE.le (↑P) ϕ.range
        Q : Subgroup K
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hle : LE.le __src✝ Q
        ⊢ Eq Q __src✝
      -/
      show Q = P.1.comap ϕ
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        P : Sylow p G
        K : Type u_2
        inst✝ : Group K
        ϕ : MonoidHom K G
        N : Subgroup G
        hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
        h : LE.le (↑P) ϕ.range
        Q : Subgroup K
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hle : LE.le __src✝ Q
        ⊢ Eq Q (Subgroup.comap ϕ ↑P)
      -/
      rw [← P.3 (hQ.map ϕ) (le_trans (ge_of_eq (map_comap_eq_self h)) (map_mono hle))]
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        P : Sylow p G
        K : Type u_2
        inst✝ : Group K
        ϕ : MonoidHom K G
        N : Subgroup G
        hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
        h : LE.le (↑P) ϕ.range
        Q : Subgroup K
        hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
        hle : LE.le __src✝ Q
        ⊢ Eq Q (Subgroup.comap ϕ (Subgroup.map ϕ Q))
      -/
      exact (comap_map_eq_self ((P.1.ker_le_comap ϕ).trans hle)).symm }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_comapOfKerIsPGroup (hϕ : IsPGroup p ϕ.ker) (h : P ≤ ϕ.range) :
    P.comapOfKerIsPGroup ϕ hϕ h = P.comap ϕ :=
  rfl


/-- The preimage of a Sylow subgroup under an injective homomorphism is a Sylow subgroup. -/
def comapOfInjective (hϕ : Function.Injective ϕ) (h : P ≤ ϕ.range) : Sylow p K :=
  P.comapOfKerIsPGroup ϕ (IsPGroup.ker_isPGroup_of_injective hϕ) h


@[simp]
theorem coe_comapOfInjective (hϕ : Function.Injective ϕ) (h : P ≤ ϕ.range) :
    P.comapOfInjective ϕ hϕ h = P.comap ϕ :=
  rfl


/-- A sylow subgroup of G is also a sylow subgroup of a subgroup of G. -/
protected def subtype (h : P ≤ N) : Sylow p N :=
                                                         /-
                                                           p : Nat
                                                           G : Type u_1
                                                           inst✝¹ : Group G
                                                           P : Sylow p G
                                                           K : Type u_2
                                                           inst✝ : Group K
                                                           ϕ : MonoidHom K G
                                                           N : Subgroup G
                                                           h : LE.le (↑P) N
                                                           ⊢ LE.le (↑P) N.subtype.range
                                                         -/
  P.comapOfInjective N.subtype Subtype.coe_injective (by rwa [range_subtype])
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem coe_subtype (h : P ≤ N) : P.subtype h = subgroupOf P N :=
  rfl


theorem subtype_injective {P Q : Sylow p G} {hP : P ≤ N} {hQ : Q ≤ N}
    (h : P.subtype hP = Q.subtype hQ) : P = Q := by
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    P Q : Sylow p G
    hP : LE.le (↑P) N
    hQ : LE.le (↑Q) N
    h : Eq (P.subtype hP) (Q.subtype hQ)
    ⊢ Eq P Q
  -/
  rw [SetLike.ext_iff] at h ⊢
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    N : Subgroup G
    P Q : Sylow p G
    hP : LE.le (↑P) N
    hQ : LE.le (↑Q) N
    h : ∀ (x : Subtype fun x => Membership.mem N x), Iff (Membership.mem (P.subtyp …
    ⊢ ∀ (x : G), Iff (Membership.mem P x) (Membership.mem Q x)
  -/
  exact fun g => ⟨fun hg => (h ⟨g, hP hg⟩).mp hg, fun hg => (h ⟨g, hQ hg⟩).mpr hg⟩
  /-
    🎉 no goals
  -/


/-- A generalization of **Sylow's first theorem**.
  Every `p`-subgroup is contained in a Sylow `p`-subgroup. -/
theorem IsPGroup.exists_le_sylow {P : Subgroup G} (hP : IsPGroup p P) : ∃ Q : Sylow p G, P ≤ Q :=
  Exists.elim
    (zorn_le_nonempty₀ { Q : Subgroup G | IsPGroup p Q }
      (fun c hc1 hc2 Q hQ =>
        ⟨{  carrier := ⋃ R : c, R
            one_mem' := ⟨Q, ⟨⟨Q, hQ⟩, rfl⟩, Q.one_mem⟩
            inv_mem' := fun {_} ⟨_, ⟨R, rfl⟩, hg⟩ => ⟨R, ⟨R, rfl⟩, R.1.inv_mem hg⟩
            mul_mem' := fun {_} _ ⟨_, ⟨R, rfl⟩, hg⟩ ⟨_, ⟨S, rfl⟩, hh⟩ =>
              (hc2.total R.2 S.2).elim (fun T => ⟨S, ⟨S, rfl⟩, S.1.mul_mem (T hg) hh⟩) fun T =>
                ⟨R, ⟨R, rfl⟩, R.1.mul_mem hg (T hh)⟩ },
          fun ⟨g, _, ⟨S, rfl⟩, hg⟩ => by
          /-
            p : Nat
            G : Type u_1
            inst✝ : Group G
            P : Subgroup G
            hP : IsPGroup p (Subtype fun x => Membership.mem P x)
            c : Set (Subgroup G)
            hc1 : HasSubset.Subset c (setOf fun Q => IsPGroup p (Subtype fun x => Membersh …
            hc2 : IsChain (fun x1 x2 => LE.le x1 x2) c
            Q : Subgroup G
            hQ : Membership.mem c Q
            x✝ : Subtype fun x => Membership.mem { carrier := Set.iUnion fun R => ↑↑R, mul …
            g : G
            S : ↑c
            hg : Membership.mem ((fun R => ↑↑R) S) g
            ⊢ Exists fun k => Eq (HPow.hPow ⟨g, ⋯⟩ (HPow.hPow p k)) 1
          -/
          refine Exists.imp (fun k hk => ?_) (hc1 S.2 ⟨g, hg⟩)
          /-
            p : Nat
            G : Type u_1
            inst✝ : Group G
            P : Subgroup G
            hP : IsPGroup p (Subtype fun x => Membership.mem P x)
            c : Set (Subgroup G)
            hc1 : HasSubset.Subset c (setOf fun Q => IsPGroup p (Subtype fun x => Membersh …
            hc2 : IsChain (fun x1 x2 => LE.le x1 x2) c
            Q : Subgroup G
            hQ : Membership.mem c Q
            x✝ : Subtype fun x => Membership.mem { carrier := Set.iUnion fun R => ↑↑R, mul …
            g : G
            S : ↑c
            hg : Membership.mem ((fun R => ↑↑R) S) g
            k : Nat
            hk : Eq (HPow.hPow ⟨g, hg⟩ (HPow.hPow p k)) 1
            ⊢ Eq (HPow.hPow ⟨g, ⋯⟩ (HPow.hPow p k)) 1
          -/
          rwa [Subtype.ext_iff, coe_pow] at hk ⊢, fun M hM _ hg => ⟨M, ⟨⟨M, hM⟩, rfl⟩, hg⟩⟩)
          /-
            🎉 no goals
          -/
      P hP)
    fun {Q} h => ⟨⟨Q, h.2.prop, h.2.eq_of_ge⟩, h.1⟩


instance nonempty : Nonempty (Sylow p G) :=
  nonempty_of_exists IsPGroup.of_bot.exists_le_sylow


noncomputable instance inhabited : Inhabited (Sylow p G) :=
  Classical.inhabited_of_nonempty nonempty


theorem exists_comap_eq_of_ker_isPGroup {H : Type*} [Group H] (P : Sylow p H) {f : H →* G}
    (hf : IsPGroup p f.ker) : ∃ Q : Sylow p G, Q.comap f = P :=
  Exists.imp (fun Q hQ => P.3 (Q.2.comap_of_ker_isPGroup f hf) (map_le_iff_le_comap.mp hQ))
    (P.2.map f).exists_le_sylow


theorem exists_comap_eq_of_injective {H : Type*} [Group H] (P : Sylow p H) {f : H →* G}
    (hf : Function.Injective f) : ∃ Q : Sylow p G, Q.comap f = P :=
  P.exists_comap_eq_of_ker_isPGroup (IsPGroup.ker_isPGroup_of_injective hf)


theorem exists_comap_subtype_eq {H : Subgroup G} (P : Sylow p H) :
    ∃ Q : Sylow p G, Q.comap H.subtype = P :=
  P.exists_comap_eq_of_injective Subtype.coe_injective


/-- If the kernel of `f : H →* G` is a `p`-group,
  then `Finite (Sylow p G)` implies `Finite (Sylow p H)`. -/
theorem finite_of_ker_is_pGroup {H : Type*} [Group H] {f : H →* G}
    (hf : IsPGroup p f.ker) [Finite (Sylow p G)] : Finite (Sylow p H) :=
  let h_exists := fun P : Sylow p H => P.exists_comap_eq_of_ker_isPGroup hf
  let g : Sylow p H → Sylow p G := fun P => Classical.choose (h_exists P)
  have hg : ∀ P : Sylow p H, (g P).1.comap f = P := fun P => Classical.choose_spec (h_exists P)
                                             /-
                                               p : Nat
                                               G : Type u_1
                                               inst✝² : Group G
                                               H : Type u_2
                                               inst✝¹ : Group H
                                               f : MonoidHom H G
                                               hf : IsPGroup p (Subtype fun x => Membership.mem f.ker x)
                                               inst✝ : Finite (Sylow p G)
                                               h_exists : ∀ (P : Sylow p H), Exists fun Q => Eq (Subgroup.comap f ↑Q) ↑P := f …
                                               g : Sylow p H → Sylow p G := fun P => Classical.choose ⋯
                                               hg : ∀ (P : Sylow p H), Eq (Subgroup.comap f ↑(g P)) ↑P
                                               P Q : Sylow p H
                                               h : Eq (g P) (g Q)
                                               ⊢ Eq ↑P ↑Q
                                             -/
  Finite.of_injective g fun P Q h => ext (by rw [← hg, h]; exact (h_exists Q).choose_spec)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- If `f : H →* G` is injective, then `Finite (Sylow p G)` implies `Finite (Sylow p H)`. -/
theorem finite_of_injective {H : Type*} [Group H] {f : H →* G}
    (hf : Function.Injective f) [Finite (Sylow p G)] : Finite (Sylow p H) :=
  finite_of_ker_is_pGroup (IsPGroup.ker_isPGroup_of_injective hf)


/-- If `H` is a subgroup of `G`, then `Finite (Sylow p G)` implies `Finite (Sylow p H)`. -/
instance (H : Subgroup G) [Finite (Sylow p G)] : Finite (Sylow p H) :=
  finite_of_injective H.subtype_injective


/-- `Subgroup.pointwiseMulAction` preserves Sylow subgroups. -/
instance pointwiseMulAction {α : Type*} [Group α] [MulDistribMulAction α G] :
    MulAction α (Sylow p G) where
  smul g P :=
    ⟨g • P.toSubgroup, P.2.map _, fun {Q} hQ hS =>
      inv_smul_eq_iff.mp
        (P.3 (hQ.map _) fun s hs =>
          (congr_arg (· ∈ g⁻¹ • Q) (inv_smul_smul g s)).mp
            (smul_mem_pointwise_smul (g • s) g⁻¹ Q (hS (smul_mem_pointwise_smul s g P hs))))⟩
  one_smul P := ext (one_smul α P.toSubgroup)
  mul_smul g h P := ext (mul_smul g h P.toSubgroup)


theorem pointwise_smul_def {α : Type*} [Group α] [MulDistribMulAction α G] {g : α}
    {P : Sylow p G} : ↑(g • P) = g • (P : Subgroup G) :=
  rfl


instance mulAction : MulAction G (Sylow p G) :=
  compHom _ MulAut.conj


theorem smul_def {g : G} {P : Sylow p G} : g • P = MulAut.conj g • P :=
  rfl


theorem coe_subgroup_smul {g : G} {P : Sylow p G} :
    ↑(g • P) = MulAut.conj g • (P : Subgroup G) :=
  rfl


theorem coe_smul {g : G} {P : Sylow p G} : ↑(g • P) = MulAut.conj g • (P : Set G) :=
  rfl


theorem smul_le {P : Sylow p G} {H : Subgroup G} (hP : P ≤ H) (h : H) : ↑(h • P) ≤ H :=
  Subgroup.conj_smul_le_of_le hP h


theorem smul_subtype {P : Sylow p G} {H : Subgroup G} (hP : P ≤ H) (h : H) :
    h • P.subtype hP = (h • P).subtype (smul_le hP h) :=
  ext (Subgroup.conj_smul_subgroupOf hP h)


theorem smul_eq_iff_mem_normalizer {g : G} {P : Sylow p G} :
    g • P = P ↔ g ∈ P.normalizer := by
  rw [eq_comm, SetLike.ext_iff, ← inv_mem_iff (G := G) (H := normalizer P.toSubgroup),
      mem_normalizer_iff, inv_inv]
  exact
    forall_congr' fun h =>
      iff_congr Iff.rfl
        ⟨fun ⟨a, b, c⟩ => c ▸ by simpa [mul_assoc] using b,
          fun hh => ⟨(MulAut.conj g)⁻¹ h, hh, MulAut.apply_inv_self G (MulAut.conj g) h⟩⟩


theorem smul_eq_of_normal {g : G} {P : Sylow p G} [h : P.Normal] :
                    /-
                      p : Nat
                      G : Type u_1
                      inst✝ : Group G
                      g : G
                      P : Sylow p G
                      h : (↑P).Normal
                      ⊢ Eq (HSMul.hSMul g P) P
                    -/
    g • P = P := by simp only [smul_eq_iff_mem_normalizer, P.normalizer_eq_top, mem_top]
                    /-
                      🎉 no goals
                    -/


theorem Subgroup.sylow_mem_fixedPoints_iff (H : Subgroup G) {P : Sylow p G} :
    P ∈ fixedPoints H (Sylow p G) ↔ H ≤ P.normalizer := by
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    P : Sylow p G
    ⊢ Iff (Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem  …
  -/
  simp_rw [SetLike.le_def, ← Sylow.smul_eq_iff_mem_normalizer]; exact Subtype.forall
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem IsPGroup.inf_normalizer_sylow {P : Subgroup G} (hP : IsPGroup p P) (Q : Sylow p G) :
    P ⊓ Q.normalizer = P ⊓ Q :=
  le_antisymm
    (le_inf inf_le_left
      (sup_eq_right.mp
        (Q.3 (hP.to_inf_left.to_sup_of_normal_right' Q.2 inf_le_right) le_sup_right)))
    (inf_le_inf_left P le_normalizer)


theorem IsPGroup.sylow_mem_fixedPoints_iff {P : Subgroup G} (hP : IsPGroup p P) {Q : Sylow p G} :
    Q ∈ fixedPoints P (Sylow p G) ↔ P ≤ Q := by
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    Q : Sylow p G
    ⊢ Iff (Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem  …
  -/
  rw [P.sylow_mem_fixedPoints_iff, ← inf_eq_left, hP.inf_normalizer_sylow, inf_eq_left]
  /-
    🎉 no goals
  -/


/-- A generalization of **Sylow's second theorem**.
  If the number of Sylow `p`-subgroups is finite, then all Sylow `p`-subgroups are conjugate. -/
instance Sylow.isPretransitive_of_finite [hp : Fact p.Prime] [Finite (Sylow p G)] :
    IsPretransitive G (Sylow p G) :=
  ⟨fun P Q => by
    classical
      have H := fun {R : Sylow p G} {S : orbit G P} =>
        calc
          S ∈ fixedPoints R (orbit G P) ↔ S.1 ∈ fixedPoints R (Sylow p G) :=
            forall_congr' fun a => Subtype.ext_iff
          _ ↔ R.1 ≤ S := R.2.sylow_mem_fixedPoints_iff
          _ ↔ S.1.1 = R := ⟨fun h => R.3 S.1.2 h, ge_of_eq⟩
      suffices Set.Nonempty (fixedPoints Q (orbit G P)) by
        exact Exists.elim this fun R hR => by
          rw [← Sylow.ext (H.mp hR)]
          exact R.2
      apply Q.2.nonempty_fixed_point_of_prime_not_dvd_card
      refine fun h => hp.out.not_dvd_one (Nat.modEq_zero_iff_dvd.mp ?_)
      calc
        1 = Nat.card (fixedPoints P (orbit G P)) := ?_
        _ ≡ Nat.card (orbit G P) [MOD p] := (P.2.card_modEq_card_fixedPoints (orbit G P)).symm
        _ ≡ 0 [MOD p] := Nat.modEq_zero_iff_dvd.mpr h
      rw [← Nat.card_unique (α := ({⟨P, mem_orbit_self P⟩} : Set (orbit G P))), eq_comm]
      congr
      rw [Set.eq_singleton_iff_unique_mem]
      exact ⟨H.mpr rfl, fun R h => Subtype.ext (Sylow.ext (H.mp h))⟩⟩


/-- A generalization of **Sylow's third theorem**.
  If the number of Sylow `p`-subgroups is finite, then it is congruent to `1` modulo `p`. -/
theorem card_sylow_modEq_one [Fact p.Prime] [Finite (Sylow p G)] :
    Nat.card (Sylow p G) ≡ 1 [MOD p] := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    ⊢ p.ModEq (Nat.card (Sylow p G)) 1
  -/
  refine Sylow.nonempty.elim fun P : Sylow p G => ?_
  have : fixedPoints P.1 (Sylow p G) = {P} :=
    Set.ext fun Q : Sylow p G =>
      calc
        Q ∈ fixedPoints P (Sylow p G) ↔ P.1 ≤ Q := P.2.sylow_mem_fixedPoints_iff
        _ ↔ Q.1 = P.1 := ⟨P.3 Q.2, ge_of_eq⟩
        _ ↔ Q ∈ {P} := Sylow.ext_iff.symm.trans Set.mem_singleton_iff.symm
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    this : Eq (MulAction.fixedPoints (Subtype fun x => Membership.mem (↑P) x) (Syl …
    ⊢ p.ModEq (Nat.card (Sylow p G)) 1
  -/
  have : Nat.card (fixedPoints P.1 (Sylow p G)) = 1 := by simp [this]
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    this✝ : Eq (MulAction.fixedPoints (Subtype fun x => Membership.mem (↑P) x) (Sy …
    this : Eq (Nat.card ↑(MulAction.fixedPoints (Subtype fun x => Membership.mem ( …
    ⊢ p.ModEq (Nat.card (Sylow p G)) 1
  -/
  exact (P.2.card_modEq_card_fixedPoints (Sylow p G)).trans (by rw [this])
  /-
    🎉 no goals
  -/


theorem not_dvd_card_sylow [hp : Fact p.Prime] [Finite (Sylow p G)] : ¬p ∣ Nat.card (Sylow p G) :=
  fun h =>
  hp.1.ne_one
    (Nat.dvd_one.mp
      ((Nat.modEq_iff_dvd' zero_le_one).mp
        ((Nat.modEq_zero_iff_dvd.mpr h).symm.trans (card_sylow_modEq_one p G))))


/-- Sylow subgroups are isomorphic -/
nonrec def equivSMul (P : Sylow p G) (g : G) : P ≃* (g • P : Sylow p G) :=
  equivSMul (MulAut.conj g) P.toSubgroup


/-- Sylow subgroups are isomorphic -/
noncomputable def equiv [Fact p.Prime] [Finite (Sylow p G)] (P Q : Sylow p G) : P ≃* Q := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P Q : Sylow p G
    ⊢ MulEquiv (Subtype fun x => Membership.mem (↑P) x) (Subtype fun x => Membersh …
  -/
  rw [← Classical.choose_spec (exists_smul_eq G P Q)]
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P Q : Sylow p G
    ⊢ MulEquiv (Subtype fun x => Membership.mem (↑P) x) (Subtype fun x => Membersh …
  -/
  exact P.equivSMul (Classical.choose (exists_smul_eq G P Q))
  /-
    🎉 no goals
  -/


@[simp]
theorem orbit_eq_top [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) : orbit G P = ⊤ :=
  top_le_iff.mp fun Q _ => exists_smul_eq G P Q


theorem stabilizer_eq_normalizer (P : Sylow p G) :
    stabilizer G P = P.normalizer := by
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    P : Sylow p G
    ⊢ Eq (MulAction.stabilizer G P) (↑P).normalizer
  -/
  ext; simp [smul_eq_iff_mem_normalizer]
       /-
         🎉 no goals
       -/


theorem conj_eq_normalizer_conj_of_mem_centralizer [Fact p.Prime] [Finite (Sylow p G)]
    (P : Sylow p G) (x g : G) (hx : x ∈ centralizer P)
    (hy : g⁻¹ * x * g ∈ centralizer P) :
    ∃ n ∈ P.normalizer, g⁻¹ * x * g = n⁻¹ * x * n := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    x g : G
    hx : Membership.mem (Subgroup.centralizer ↑P) x
    hy : Membership.mem (Subgroup.centralizer ↑P) (HMul.hMul (HMul.hMul (Inv.inv g …
    ⊢ Exists fun n => And (Membership.mem (↑P).normalizer n) (Eq (HMul.hMul (HMul. …
  -/
  have h1 : P ≤ centralizer (zpowers x : Set G) := by rwa [le_centralizer_iff, zpowers_le]
  have h2 : ↑(g • P) ≤ centralizer (zpowers x : Set G) := by
    rw [le_centralizer_iff, zpowers_le]
    rintro - ⟨z, hz, rfl⟩
    specialize hy z hz
    rwa [← mul_assoc, ← eq_mul_inv_iff_mul_eq, mul_assoc, mul_assoc, mul_assoc, ← mul_assoc,
      eq_inv_mul_iff_mul_eq, ← mul_assoc, ← mul_assoc] at hy
  obtain ⟨h, hh⟩ :=
    exists_smul_eq (centralizer (zpowers x : Set G)) ((g • P).subtype h2) (P.subtype h1)
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    x g : G
    hx : Membership.mem (Subgroup.centralizer ↑P) x
    hy : Membership.mem (Subgroup.centralizer ↑P) (HMul.hMul (HMul.hMul (Inv.inv g …
    h1 : LE.le (↑P) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h2 : LE.le (↑(HSMul.hSMul g P)) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h : Subtype fun x_1 => Membership.mem (Subgroup.centralizer ↑(Subgroup.zpowers …
    hh : Eq (HSMul.hSMul h ((HSMul.hSMul g P).subtype h2)) (P.subtype h1)
    ⊢ Exists fun n => And (Membership.mem (↑P).normalizer n) (Eq (HMul.hMul (HMul. …
  -/
  simp_rw [smul_subtype, Subgroup.smul_def, smul_smul] at hh
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    x g : G
    hx : Membership.mem (Subgroup.centralizer ↑P) x
    hy : Membership.mem (Subgroup.centralizer ↑P) (HMul.hMul (HMul.hMul (Inv.inv g …
    h1 : LE.le (↑P) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h2 : LE.le (↑(HSMul.hSMul g P)) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h : Subtype fun x_1 => Membership.mem (Subgroup.centralizer ↑(Subgroup.zpowers …
    hh : Eq ((HSMul.hSMul (HMul.hMul (↑h) g) P).subtype ⋯) (P.subtype h1)
    ⊢ Exists fun n => And (Membership.mem (↑P).normalizer n) (Eq (HMul.hMul (HMul. …
  -/
  refine ⟨h * g, smul_eq_iff_mem_normalizer.mp (subtype_injective hh), ?_⟩
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    x g : G
    hx : Membership.mem (Subgroup.centralizer ↑P) x
    hy : Membership.mem (Subgroup.centralizer ↑P) (HMul.hMul (HMul.hMul (Inv.inv g …
    h1 : LE.le (↑P) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h2 : LE.le (↑(HSMul.hSMul g P)) (Subgroup.centralizer ↑(Subgroup.zpowers x))
    h : Subtype fun x_1 => Membership.mem (Subgroup.centralizer ↑(Subgroup.zpowers …
    hh : Eq ((HSMul.hSMul (HMul.hMul (↑h) g) P).subtype ⋯) (P.subtype h1)
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv g) x) g) (HMul.hMul (HMul.hMul (Inv.inv (H …
  -/
  rw [← mul_assoc, Commute.right_comm (h.prop x (mem_zpowers x)), mul_inv_rev, inv_mul_cancel_right]
  /-
    🎉 no goals
  -/


theorem conj_eq_normalizer_conj_of_mem [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G)
    [_hP : P.IsCommutative] (x g : G) (hx : x ∈ P) (hy : g⁻¹ * x * g ∈ P) :
    ∃ n ∈ P.normalizer, g⁻¹ * x * g = n⁻¹ * x * n :=
  P.conj_eq_normalizer_conj_of_mem_centralizer x g
    (P.le_centralizer hx) (P.le_centralizer hy)


/-- Sylow `p`-subgroups are in bijection with cosets of the normalizer of a Sylow `p`-subgroup -/
noncomputable def equivQuotientNormalizer [Fact p.Prime] [Finite (Sylow p G)]
    (P : Sylow p G) : Sylow p G ≃ G ⧸ P.normalizer :=
  calc
    Sylow p G ≃ (⊤ : Set (Sylow p G)) := (Equiv.Set.univ (Sylow p G)).symm
    _ ≃ orbit G P := Equiv.setCongr P.orbit_eq_top.symm
    _ ≃ G ⧸ stabilizer G P := orbitEquivQuotientStabilizer G P
                               /-
                                 p : Nat
                                 G : Type u_1
                                 inst✝² : Group G
                                 inst✝¹ : Fact (Nat.Prime p)
                                 inst✝ : Finite (Sylow p G)
                                 P : Sylow p G
                                 ⊢ Equiv (HasQuotient.Quotient G (MulAction.stabilizer G P)) (HasQuotient.Quoti …
                               -/
    _ ≃ G ⧸ P.normalizer := by rw [P.stabilizer_eq_normalizer]
                               /-
                                 🎉 no goals
                               -/


instance [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) :
    Finite (G ⧸ P.normalizer) :=
  Finite.of_equiv (Sylow p G) P.equivQuotientNormalizer


theorem card_eq_card_quotient_normalizer [Fact p.Prime] [Finite (Sylow p G)]
    (P : Sylow p G) : Nat.card (Sylow p G) = Nat.card (G ⧸ P.normalizer) :=
  Nat.card_congr P.equivQuotientNormalizer


@[deprecated (since := "2024-11-07")]
alias _root_.card_sylow_eq_card_quotient_normalizer := card_eq_card_quotient_normalizer


theorem card_eq_index_normalizer [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) :
    Nat.card (Sylow p G) = P.normalizer.index :=
  P.card_eq_card_quotient_normalizer


@[deprecated (since := "2024-11-07")]
alias _root_.card_sylow_eq_index_normalizer := card_eq_index_normalizer


theorem card_dvd_index [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) :
    Nat.card (Sylow p G) ∣ P.index :=
  ((congr_arg _ P.card_eq_index_normalizer).mp dvd_rfl).trans
    (index_dvd_of_le le_normalizer)


@[deprecated (since := "2024-11-07")]
alias _root_.card_sylow_dvd_index := card_dvd_index


/-- Auxiliary lemma for `Sylow.not_dvd_index` which is strictly stronger. -/
private theorem not_dvd_index_aux [hp : Fact p.Prime] (P : Sylow p G) [P.Normal]
    [P.FiniteIndex] : ¬ p ∣ P.index := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    ⊢ Not (Dvd.dvd p (↑P).index)
  -/
  intro h
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h : Dvd.dvd p (↑P).index
    ⊢ False
  -/
  rw [P.index_eq_card] at h
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h : Dvd.dvd p (Nat.card (HasQuotient.Quotient G ↑P))
    ⊢ False
  -/
  obtain ⟨x, hx⟩ := exists_prime_orderOf_dvd_card' (G := G ⧸ (P : Subgroup G)) p h
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h : Dvd.dvd p (Nat.card (HasQuotient.Quotient G ↑P))
    x : HasQuotient.Quotient G ↑P
    hx : Eq (orderOf x) p
    ⊢ False
  -/
  have h := IsPGroup.of_card (((Nat.card_zpowers x).trans hx).trans (pow_one p).symm)
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h✝ : Dvd.dvd p (Nat.card (HasQuotient.Quotient G ↑P))
    x : HasQuotient.Quotient G ↑P
    hx : Eq (orderOf x) p
    h : IsPGroup p (Subtype fun x_1 => Membership.mem (Subgroup.zpowers x) x_1)
    ⊢ False
  -/
  let Q := (zpowers x).comap (QuotientGroup.mk' (P : Subgroup G))
  have hQ : IsPGroup p Q := by
    apply h.comap_of_ker_isPGroup
    rw [QuotientGroup.ker_mk']
    exact P.2
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h✝ : Dvd.dvd p (Nat.card (HasQuotient.Quotient G ↑P))
    x : HasQuotient.Quotient G ↑P
    hx : Eq (orderOf x) p
    h : IsPGroup p (Subtype fun x_1 => Membership.mem (Subgroup.zpowers x) x_1)
    Q : Subgroup G := Subgroup.comap (QuotientGroup.mk' ↑P) (Subgroup.zpowers x)
    hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
    ⊢ False
  -/
  replace hp := mt orderOf_eq_one_iff.mpr (ne_of_eq_of_ne hx hp.1.ne_one)
  rw [← zpowers_eq_bot, ← Ne, ← bot_lt_iff_ne_bot, ←
    comap_lt_comap_of_surjective (QuotientGroup.mk'_surjective _), MonoidHom.comap_bot,
    QuotientGroup.ker_mk'] at hp
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    P : Sylow p G
    inst✝¹ : (↑P).Normal
    inst✝ : (↑P).FiniteIndex
    h✝ : Dvd.dvd p (Nat.card (HasQuotient.Quotient G ↑P))
    x : HasQuotient.Quotient G ↑P
    hx : Eq (orderOf x) p
    h : IsPGroup p (Subtype fun x_1 => Membership.mem (Subgroup.zpowers x) x_1)
    Q : Subgroup G := Subgroup.comap (QuotientGroup.mk' ↑P) (Subgroup.zpowers x)
    hQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
    hp : LT.lt (↑P) (Subgroup.comap (QuotientGroup.mk' ↑P) (Subgroup.zpowers x))
    ⊢ False
  -/
  exact hp.ne' (P.3 hQ hp.le)
  /-
    🎉 no goals
  -/


/-- A Sylow p-subgroup has index indivisible by `p`, assuming [N(P) : P] < ∞. -/
theorem not_dvd_index' [hp : Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G)
    (hP : P.relindex P.normalizer ≠ 0) : ¬ p ∣ P.index := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    hP : Ne ((↑P).relindex (↑P).normalizer) 0
    ⊢ Not (Dvd.dvd p (↑P).index)
  -/
  rw [← relindex_mul_index le_normalizer, ← card_eq_index_normalizer]
  haveI : (P.subtype le_normalizer).Normal :=
    Subgroup.normal_in_normalizer
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    hP : Ne ((↑P).relindex (↑P).normalizer) 0
    this : (↑(P.subtype ⋯)).Normal
    ⊢ Not (Dvd.dvd p (HMul.hMul ((↑P).relindex (↑P).normalizer) (Nat.card (Sylow p …
  -/
  haveI : (P.subtype le_normalizer).FiniteIndex := ⟨hP⟩
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    hP : Ne ((↑P).relindex (↑P).normalizer) 0
    this✝ : (↑(P.subtype ⋯)).Normal
    this : (↑(P.subtype ⋯)).FiniteIndex
    ⊢ Not (Dvd.dvd p (HMul.hMul ((↑P).relindex (↑P).normalizer) (Nat.card (Sylow p …
  -/
  replace hP := not_dvd_index_aux (P.subtype le_normalizer)
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    this✝ : (↑(P.subtype ⋯)).Normal
    this : (↑(P.subtype ⋯)).FiniteIndex
    hP : Not (Dvd.dvd p (↑(P.subtype ⋯)).index)
    ⊢ Not (Dvd.dvd p (HMul.hMul ((↑P).relindex (↑P).normalizer) (Nat.card (Sylow p …
  -/
  exact hp.1.not_dvd_mul hP (not_dvd_card_sylow p G)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-03")]
alias _root_.not_dvd_index_sylow := not_dvd_index'


/-- A Sylow p-subgroup has index indivisible by `p`. -/
theorem not_dvd_index [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) [P.FiniteIndex] :
    ¬ p ∣ P.index :=
  P.not_dvd_index' Nat.card_pos.ne'


@[deprecated (since := "2024-11-03")]
alias _root_.not_dvd_index_sylow' := not_dvd_index


/-- Surjective group homomorphisms map Sylow subgroups to Sylow subgroups. -/
def mapSurjective [Fact p.Prime] (P : Sylow p G) : Sylow p G' :=
  { P.1.map f with
    isPGroup' := P.2.map f
    is_maximal' := fun hQ hPQ ↦ ((P.2.map f).toSylow
      (fun h ↦ P.not_dvd_index (h.trans (P.index_map_dvd hf)))).3 hQ hPQ }


@[simp] theorem coe_mapSurjective [Fact p.Prime] (P : Sylow p G) : P.mapSurjective hf = P.map f :=
  rfl


theorem mapSurjective_surjective (p : ℕ) [Fact p.Prime] :
    Function.Surjective (Sylow.mapSurjective hf : Sylow p G → Sylow p G') := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Function.Surjective (Sylow.mapSurjective hf)
  -/
  have : Finite G' := Finite.of_surjective f hf
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    ⊢ Function.Surjective (Sylow.mapSurjective hf)
  -/
  intro P
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    ⊢ Exists fun a => Eq (Sylow.mapSurjective hf a) P
  -/
  let Q₀ : Sylow p (P.comap f) := Sylow.nonempty.some
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    ⊢ Exists fun a => Eq (Sylow.mapSurjective hf a) P
  -/
  let Q : Subgroup G := Q₀.map (P.comap f).subtype
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    Q : Subgroup G := Subgroup.map (Subgroup.comap f ↑P).subtype ↑Q₀
    ⊢ Exists fun a => Eq (Sylow.mapSurjective hf a) P
  -/
  have hPQ : Q.map f ≤ P := Subgroup.map_le_iff_le_comap.mpr (Subgroup.map_subtype_le Q₀.1)
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    Q : Subgroup G := Subgroup.map (Subgroup.comap f ↑P).subtype ↑Q₀
    hPQ : LE.le (Subgroup.map f Q) ↑P
    ⊢ Exists fun a => Eq (Sylow.mapSurjective hf a) P
  -/
  have hpQ : IsPGroup p Q := Q₀.2.map (P.comap f).subtype
  have hQ : ¬ p ∣ Q.index := by
    rw [Subgroup.index_map_subtype Q₀.1, P.index_comap_of_surjective hf]
    exact Nat.Prime.not_dvd_mul Fact.out Q₀.not_dvd_index P.not_dvd_index
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    Q : Subgroup G := Subgroup.map (Subgroup.comap f ↑P).subtype ↑Q₀
    hPQ : LE.le (Subgroup.map f Q) ↑P
    hpQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
    hQ : Not (Dvd.dvd p Q.index)
    ⊢ Exists fun a => Eq (Sylow.mapSurjective hf a) P
  -/
  use hpQ.toSylow hQ
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    Q : Subgroup G := Subgroup.map (Subgroup.comap f ↑P).subtype ↑Q₀
    hPQ : LE.le (Subgroup.map f Q) ↑P
    hpQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
    hQ : Not (Dvd.dvd p Q.index)
    ⊢ Eq (Sylow.mapSurjective hf (hpQ.toSylow hQ)) P
  -/
  rw [Sylow.ext_iff, Sylow.coe_mapSurjective, eq_comm]
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    G' : Type u_2
    inst✝¹ : Group G'
    f : MonoidHom G G'
    hf : Function.Surjective ⇑f
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Finite G'
    P : Sylow p G'
    Q₀ : Sylow p (Subtype fun x => Membership.mem (Subgroup.comap f ↑P) x) := ⋯.some
    Q : Subgroup G := Subgroup.map (Subgroup.comap f ↑P).subtype ↑Q₀
    hPQ : LE.le (Subgroup.map f Q) ↑P
    hpQ : IsPGroup p (Subtype fun x => Membership.mem Q x)
    hQ : Not (Dvd.dvd p Q.index)
    ⊢ Eq (↑P) (Subgroup.map f ↑(hpQ.toSylow hQ))
  -/
  exact ((hpQ.map f).toSylow (fun h ↦ hQ (h.trans (Q.index_map_dvd hf)))).3 P.2 hPQ
  /-
    🎉 no goals
  -/


/-- **Frattini's Argument**: If `N` is a normal subgroup of `G`, and if `P` is a Sylow `p`-subgroup
  of `N`, then `N_G(P) ⊔ N = G`. -/
theorem normalizer_sup_eq_top {p : ℕ} [Fact p.Prime] {N : Subgroup G} [N.Normal]
    [Finite (Sylow p N)] (P : Sylow p N) :
    (P.map N.subtype).normalizer ⊔ N = ⊤ := by
  /-
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    ⊢ Eq (Max.max (Subgroup.map N.subtype ↑P).normalizer N) Top.top
  -/
  refine top_le_iff.mp fun g _ => ?_
  /-
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    ⊢ Membership.mem (Max.max (Subgroup.map N.subtype ↑P).normalizer N) g
  -/
  obtain ⟨n, hn⟩ := exists_smul_eq N ((MulAut.conjNormal g : MulAut N) • P) P
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (HSMul.hSMul n (HSMul.hSMul (MulAut.conjNormal g) P)) P
    ⊢ Membership.mem (Max.max (Subgroup.map N.subtype ↑P).normalizer N) g
  -/
  rw [← inv_mul_cancel_left (↑n) g, sup_comm]
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (HSMul.hSMul n (HSMul.hSMul (MulAut.conjNormal g) P)) P
    ⊢ Membership.mem (Max.max N (Subgroup.map N.subtype ↑P).normalizer) (HMul.hMul …
  -/
  apply mul_mem_sup (N.inv_mem n.2)
  rw [smul_def, ← mul_smul, ← MulAut.conjNormal_val, ← MulAut.conjNormal.map_mul,
    Sylow.ext_iff, pointwise_smul_def, Subgroup.pointwise_smul_def] at hn
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Subgroup.map ((MulDistribMulAction.toMonoidEnd (MulAut (Subtype fun x …
    ⊢ Membership.mem (Subgroup.map N.subtype ↑P).normalizer (HMul.hMul (↑n) g)
  -/
  have : Function.Injective (MulAut.conj (n * g)).toMonoidHom := (MulAut.conj (n * g)).injective
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Subgroup.map ((MulDistribMulAction.toMonoidEnd (MulAut (Subtype fun x …
    this : Function.Injective ⇑(MulEquiv.toMonoidHom (MulAut.conj (HMul.hMul (↑n)  …
    ⊢ Membership.mem (Subgroup.map N.subtype ↑P).normalizer (HMul.hMul (↑n) g)
  -/
  refine fun x ↦ (mem_map_iff_mem this).symm.trans ?_
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Subgroup.map ((MulDistribMulAction.toMonoidEnd (MulAut (Subtype fun x …
    this : Function.Injective ⇑(MulEquiv.toMonoidHom (MulAut.conj (HMul.hMul (↑n)  …
    x : G
    ⊢ Iff (Membership.mem (Subgroup.map (MulEquiv.toMonoidHom (MulAut.conj (HMul.h …
  -/
  rw [map_map, ← congr_arg (map N.subtype) hn, map_map]
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    N : Subgroup G
    inst✝¹ : N.Normal
    inst✝ : Finite (Sylow p (Subtype fun x => Membership.mem N x))
    P : Sylow p (Subtype fun x => Membership.mem N x)
    g : G
    x✝ : Membership.mem Top.top g
    n : Subtype fun x => Membership.mem N x
    hn : Eq (Subgroup.map ((MulDistribMulAction.toMonoidEnd (MulAut (Subtype fun x …
    this : Function.Injective ⇑(MulEquiv.toMonoidHom (MulAut.conj (HMul.hMul (↑n)  …
    x : G
    ⊢ Iff (Membership.mem (Subgroup.map ((MulEquiv.toMonoidHom (MulAut.conj (HMul. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **Frattini's Argument**: If `N` is a normal subgroup of `G`, and if `P` is a Sylow `p`-subgroup
  of `N`, then `N_G(P) ⊔ N = G`. -/
theorem normalizer_sup_eq_top' {p : ℕ} [Fact p.Prime] {N : Subgroup G} [N.Normal]
    [Finite (Sylow p N)] (P : Sylow p G) (hP : P ≤ N) : P.normalizer ⊔ N = ⊤ := by
  rw [← normalizer_sup_eq_top (P.subtype hP), P.coe_subtype, subgroupOf_map_subtype,
    inf_of_le_left hP]


theorem QuotientGroup.card_preimage_mk (s : Subgroup G) (t : Set (G ⧸ s)) :
    Nat.card (QuotientGroup.mk ⁻¹' t) = Nat.card s * Nat.card t := by
  /-
    G : Type u
    inst✝ : Group G
    s : Subgroup G
    t : Set (HasQuotient.Quotient G s)
    ⊢ Eq (Nat.card ↑(Set.preimage QuotientGroup.mk t)) (HMul.hMul (Nat.card (Subty …
  -/
  rw [← Nat.card_prod, Nat.card_congr (preimageMkEquivSubgroupProdSet _ _)]
  /-
    🎉 no goals
  -/


theorem mem_fixedPoints_mul_left_cosets_iff_mem_normalizer {H : Subgroup G} [Finite (H : Set G)]
    {x : G} : (x : G ⧸ H) ∈ MulAction.fixedPoints H (G ⧸ H) ↔ x ∈ normalizer H :=
  ⟨fun hx =>
    have ha : ∀ {y : G ⧸ H}, y ∈ orbit H (x : G ⧸ H) → y = x := mem_fixedPoints'.1 hx _
    (inv_mem_iff (G := G)).1
      (mem_normalizer_fintype fun n (hn : n ∈ H) =>
        have : (n⁻¹ * x)⁻¹ * x ∈ H := QuotientGroup.eq.1 (ha ⟨⟨n⁻¹, inv_mem hn⟩, rfl⟩)
        show _ ∈ H by
          /-
            G : Type u
            inst✝¹ : Group G
            H : Subgroup G
            inst✝ : Finite ↑↑H
            x : G
            hx : Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem H  …
            ha : ∀ {y : HasQuotient.Quotient G H}, Membership.mem (MulAction.orbit (Subtyp …
            n : G
            hn : Membership.mem H n
            this : Membership.mem H (HMul.hMul (Inv.inv (HMul.hMul (Inv.inv n) x)) x)
            ⊢ Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) n) (Inv.inv (Inv.inv x)))
          -/
          rw [mul_inv_rev, inv_inv] at this
          /-
            G : Type u
            inst✝¹ : Group G
            H : Subgroup G
            inst✝ : Finite ↑↑H
            x : G
            hx : Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem H  …
            ha : ∀ {y : HasQuotient.Quotient G H}, Membership.mem (MulAction.orbit (Subtyp …
            n : G
            hn : Membership.mem H n
            this : Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) n) x)
            ⊢ Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) n) (Inv.inv (Inv.inv x)))
          -/
          convert this
          /-
            case h.e'_5.h.e'_6
            G : Type u
            inst✝¹ : Group G
            H : Subgroup G
            inst✝ : Finite ↑↑H
            x : G
            hx : Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem H  …
            ha : ∀ {y : HasQuotient.Quotient G H}, Membership.mem (MulAction.orbit (Subtyp …
            n : G
            hn : Membership.mem H n
            this : Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv x) n) x)
            ⊢ Eq (Inv.inv (Inv.inv x)) x
          -/
          rw [inv_inv]),
          /-
            🎉 no goals
          -/
    fun hx : ∀ n : G, n ∈ H ↔ x * n * x⁻¹ ∈ H =>
    mem_fixedPoints'.2 fun y =>
      Quotient.inductionOn' y fun y hy =>
        QuotientGroup.eq.2
          (let ⟨⟨b, hb₁⟩, hb₂⟩ := hy
          have hb₂ : (b * x)⁻¹ * y ∈ H := QuotientGroup.eq.1 hb₂
          (inv_mem_iff (G := G)).1 <|
            (hx _).2 <|
              (mul_mem_cancel_left (inv_mem hb₁)).1 <| by
                /-
                  G : Type u
                  inst✝¹ : Group G
                  H : Subgroup G
                  inst✝ : Finite ↑↑H
                  x : G
                  hx : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
                  y✝ : HasQuotient.Quotient G H
                  y : G
                  hy : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑x) …
                  b : G
                  hb₁ : Membership.mem H b
                  hb₂✝ : Eq ((fun m => HSMul.hSMul m ↑x) ⟨b, hb₁⟩) (Quotient.mk'' y)
                  hb₂ : Membership.mem H (HMul.hMul (Inv.inv (HMul.hMul b x)) y)
                  ⊢ Membership.mem H (HMul.hMul (Inv.inv b) (HMul.hMul (HMul.hMul x (Inv.inv (HM …
                -/
                rw [hx] at hb₂; simpa [mul_inv_rev, mul_assoc] using hb₂)⟩
                                /-
                                  🎉 no goals
                                -/


/-- The fixed points of the action of `H` on its cosets correspond to `normalizer H / H`. -/
def fixedPointsMulLeftCosetsEquivQuotient (H : Subgroup G) [Finite (H : Set G)] :
    MulAction.fixedPoints H (G ⧸ H) ≃
      normalizer H ⧸ Subgroup.comap ((normalizer H).subtype : normalizer H →* G) H :=
  @subtypeQuotientEquivQuotientSubtype G (normalizer H : Set G) (_) (_)
    (MulAction.fixedPoints H (G ⧸ H))
    (fun _ => (@mem_fixedPoints_mul_left_cosets_iff_mem_normalizer _ _ _ ‹_› _).symm)
    (by
      /-
        G : Type u
        inst✝¹ : Group G
        H : Subgroup G
        inst✝ : Finite ↑↑H
        ⊢ ∀ (x y : Subtype ↑H.normalizer), Iff ((QuotientGroup.leftRel (Subgroup.comap …
      -/
      intros
      /-
        G : Type u
        inst✝¹ : Group G
        H : Subgroup G
        inst✝ : Finite ↑↑H
        x✝ y✝ : Subtype ↑H.normalizer
        ⊢ Iff ((QuotientGroup.leftRel (Subgroup.comap H.normalizer.subtype H)) x✝ y✝)  …
      -/
      unfold_projs
      /-
        G : Type u
        inst✝¹ : Group G
        H : Subgroup G
        inst✝ : Finite ↑↑H
        x✝ y✝ : Subtype ↑H.normalizer
        ⊢ Iff ((QuotientGroup.leftRel (Subgroup.comap H.normalizer.subtype H)) x✝ y✝)  …
      -/
      rw [leftRel_apply (α := normalizer H), leftRel_apply]
      /-
        G : Type u
        inst✝¹ : Group G
        H : Subgroup G
        inst✝ : Finite ↑↑H
        x✝ y✝ : Subtype ↑H.normalizer
        ⊢ Iff (Membership.mem (Subgroup.comap H.normalizer.subtype H) (HMul.hMul (Inv. …
      -/
      rfl)
      /-
        🎉 no goals
      -/


/-- If `H` is a `p`-subgroup of `G`, then the index of `H` inside its normalizer is congruent
  mod `p` to the index of `H`. -/
theorem card_quotient_normalizer_modEq_card_quotient [Finite G] {p : ℕ} {n : ℕ} [hp : Fact p.Prime]
    {H : Subgroup G} (hH : Nat.card H = p ^ n) :
    Nat.card (normalizer H ⧸ Subgroup.comap ((normalizer H).subtype : normalizer H →* G) H) ≡
      Nat.card (G ⧸ H) [MOD p] := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
    ⊢ p.ModEq (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.mem H.n …
  -/
  rw [← Nat.card_congr (fixedPointsMulLeftCosetsEquivQuotient H)]
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
    ⊢ p.ModEq (Nat.card ↑(MulAction.fixedPoints (Subtype fun x => Membership.mem H …
  -/
  exact ((IsPGroup.of_card hH).card_modEq_card_fixedPoints _).symm
  /-
    🎉 no goals
  -/


/-- If `H` is a subgroup of `G` of cardinality `p ^ n`, then the cardinality of the
  normalizer of `H` is congruent mod `p ^ (n + 1)` to the cardinality of `G`. -/
theorem card_normalizer_modEq_card [Finite G] {p : ℕ} {n : ℕ} [hp : Fact p.Prime] {H : Subgroup G}
    (hH : Nat.card H = p ^ n) : Nat.card (normalizer H) ≡ Nat.card G [MOD p ^ (n + 1)] := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
    ⊢ (HPow.hPow p (HAdd.hAdd n 1)).ModEq (Nat.card (Subtype fun x => Membership.m …
  -/
  have : H.subgroupOf (normalizer H) ≃ H := (subgroupOfEquivOfLe le_normalizer).toEquiv
  rw [card_eq_card_quotient_mul_card_subgroup H,
    card_eq_card_quotient_mul_card_subgroup (H.subgroupOf (normalizer H)), Nat.card_congr this,
    hH, pow_succ']
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
    this : Equiv (Subtype fun x => Membership.mem (H.subgroupOf H.normalizer) x) ( …
    ⊢ (HMul.hMul p (HPow.hPow p n)).ModEq (HMul.hMul (Nat.card (HasQuotient.Quotie …
  -/
  exact (card_quotient_normalizer_modEq_card_quotient hH).mul_right' _
  /-
    🎉 no goals
  -/


/-- If `H` is a `p`-subgroup but not a Sylow `p`-subgroup, then `p` divides the
  index of `H` inside its normalizer. -/
theorem prime_dvd_card_quotient_normalizer [Finite G] {p : ℕ} {n : ℕ} [Fact p.Prime]
    (hdvd : p ^ (n + 1) ∣ Nat.card G) {H : Subgroup G} (hH : Nat.card H = p ^ n) :
    p ∣ Nat.card (normalizer H ⧸ Subgroup.comap ((normalizer H).subtype : normalizer H →* G) H) :=
  let ⟨s, hs⟩ := exists_eq_mul_left_of_dvd hdvd
  have hcard : Nat.card (G ⧸ H) = s * p :=
    (mul_left_inj' (show Nat.card H ≠ 0 from Nat.card_pos.ne')).1
      (by
        /-
          G : Type u
          inst✝² : Group G
          inst✝¹ : Finite G
          p n : Nat
          inst✝ : Fact (Nat.Prime p)
          hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
          H : Subgroup G
          hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
          s : Nat
          hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
          ⊢ Eq (HMul.hMul (Nat.card (HasQuotient.Quotient G H)) (Nat.card (Subtype fun x …
        -/
        rw [← card_eq_card_quotient_mul_card_subgroup H, hH, hs, pow_succ', mul_assoc, mul_comm p])
        /-
          🎉 no goals
        -/
  have hm :
    s * p % p =
      Nat.card (normalizer H ⧸ Subgroup.comap ((normalizer H).subtype : normalizer H →* G) H) % p :=
    hcard ▸ (card_quotient_normalizer_modEq_card_quotient hH).symm
                             /-
                               G : Type u
                               inst✝² : Group G
                               inst✝¹ : Finite G
                               p n : Nat
                               inst✝ : Fact (Nat.Prime p)
                               hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
                               H : Subgroup G
                               hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
                               s : Nat
                               hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
                               hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
                               hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
                               ⊢ Eq (HMod.hMod (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
                             -/
  Nat.dvd_of_mod_eq_zero (by rwa [Nat.mod_eq_zero_of_dvd (dvd_mul_left _ _), eq_comm] at hm)
                             /-
                               🎉 no goals
                             -/


/-- If `H` is a `p`-subgroup but not a Sylow `p`-subgroup of cardinality `p ^ n`,
  then `p ^ (n + 1)` divides the cardinality of the normalizer of `H`. -/
theorem prime_pow_dvd_card_normalizer [Finite G] {p : ℕ} {n : ℕ} [_hp : Fact p.Prime]
    (hdvd : p ^ (n + 1) ∣ Nat.card G) {H : Subgroup G} (hH : Nat.card H = p ^ n) :
    p ^ (n + 1) ∣ Nat.card (normalizer H) :=
  Nat.modEq_zero_iff_dvd.1 ((card_normalizer_modEq_card hH).trans hdvd.modEq_zero_nat)


/-- If `H` is a subgroup of `G` of cardinality `p ^ n`,
  then `H` is contained in a subgroup of cardinality `p ^ (n + 1)`
  if `p ^ (n + 1)` divides the cardinality of `G` -/
theorem exists_subgroup_card_pow_succ [Finite G] {p : ℕ} {n : ℕ} [hp : Fact p.Prime]
    (hdvd : p ^ (n + 1) ∣ Nat.card G) {H : Subgroup G} (hH : Nat.card H = p ^ n) :
    ∃ K : Subgroup G, Nat.card K = p ^ (n + 1) ∧ H ≤ K :=
  let ⟨s, hs⟩ := exists_eq_mul_left_of_dvd hdvd
  have hcard : Nat.card (G ⧸ H) = s * p :=
    (mul_left_inj' (show Nat.card H ≠ 0 from Nat.card_pos.ne')).1
      (by
        /-
          G : Type u
          inst✝¹ : Group G
          inst✝ : Finite G
          p n : Nat
          hp : Fact (Nat.Prime p)
          hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
          H : Subgroup G
          hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
          s : Nat
          hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
          ⊢ Eq (HMul.hMul (Nat.card (HasQuotient.Quotient G H)) (Nat.card (Subtype fun x …
        -/
        rw [← card_eq_card_quotient_mul_card_subgroup H, hH, hs, pow_succ', mul_assoc, mul_comm p])
        /-
          🎉 no goals
        -/
  have hm : s * p % p = Nat.card (normalizer H ⧸ H.subgroupOf H.normalizer) % p :=
    Nat.card_congr (fixedPointsMulLeftCosetsEquivQuotient H) ▸
      hcard ▸ (IsPGroup.of_card hH).card_modEq_card_fixedPoints _
  have hm' : p ∣ Nat.card (normalizer H ⧸ H.subgroupOf H.normalizer) :=
                               /-
                                 G : Type u
                                 inst✝¹ : Group G
                                 inst✝ : Finite G
                                 p n : Nat
                                 hp : Fact (Nat.Prime p)
                                 hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
                                 H : Subgroup G
                                 hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
                                 s : Nat
                                 hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
                                 hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
                                 hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
                                 ⊢ Eq (HMod.hMod (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
                               -/
    Nat.dvd_of_mod_eq_zero (by rwa [Nat.mod_eq_zero_of_dvd (dvd_mul_left _ _), eq_comm] at hm)
                               /-
                                 🎉 no goals
                               -/
  let ⟨x, hx⟩ := @exists_prime_orderOf_dvd_card' _ (QuotientGroup.Quotient.group _) _ _ hp hm'
  have hequiv : H ≃ H.subgroupOf H.normalizer := (subgroupOfEquivOfLe le_normalizer).symm.toEquiv
  ⟨Subgroup.map (normalizer H).subtype
      (Subgroup.comap (mk' (H.subgroupOf H.normalizer)) (zpowers x)), by
    show Nat.card (Subgroup.map H.normalizer.subtype
              (comap (mk' (H.subgroupOf H.normalizer)) (Subgroup.zpowers x))) = p ^ (n + 1)
    suffices Nat.card (Subtype.val ''
              (Subgroup.comap (mk' (H.subgroupOf H.normalizer)) (zpowers x) : Set H.normalizer)) =
        p ^ (n + 1)
      by convert this using 2
    rw [Nat.card_image_of_injective Subtype.val_injective
        (Subgroup.comap (mk' (H.subgroupOf H.normalizer)) (zpowers x) : Set H.normalizer),
      pow_succ, ← hH, Nat.card_congr hequiv, ← hx, ← Nat.card_zpowers, ←
      Nat.card_prod]
    exact Nat.card_congr
      (preimageMkEquivSubgroupProdSet (H.subgroupOf H.normalizer) (zpowers x)), by
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      ⊢ LE.le H (Subgroup.map H.normalizer.subtype (Subgroup.comap (QuotientGroup.mk …
    -/
    intro y hy
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      y : G
      hy : Membership.mem H y
      ⊢ Membership.mem (Subgroup.map H.normalizer.subtype (Subgroup.comap (QuotientG …
    -/
    simp only [exists_prop, Subgroup.coeSubtype, mk'_apply, Subgroup.mem_map, Subgroup.mem_comap]
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      y : G
      hy : Membership.mem H y
      ⊢ Exists fun x_1 => And (Membership.mem (Subgroup.zpowers x) ↑x_1) (Eq (↑x_1) y)
    -/
    refine ⟨⟨y, le_normalizer hy⟩, ⟨0, ?_⟩, rfl⟩
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      y : G
      hy : Membership.mem H y
      ⊢ Eq ((fun x_1 => HPow.hPow x x_1) 0) ↑⟨y, ⋯⟩
    -/
    dsimp only
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      y : G
      hy : Membership.mem H y
      ⊢ Eq (HPow.hPow x 0) ↑⟨y, ⋯⟩
    -/
    rw [zpow_zero, eq_comm, QuotientGroup.eq_one_iff]
    /-
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      p n : Nat
      hp : Fact (Nat.Prime p)
      hdvd : Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) (Nat.card G)
      H : Subgroup G
      hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
      s : Nat
      hs : Eq (Nat.card G) (HMul.hMul s (HPow.hPow p (HAdd.hAdd n 1)))
      hcard : Eq (Nat.card (HasQuotient.Quotient G H)) (HMul.hMul s p)
      hm : Eq (HMod.hMod (HMul.hMul s p) p) (HMod.hMod (Nat.card (HasQuotient.Quotie …
      hm' : Dvd.dvd p (Nat.card (HasQuotient.Quotient (Subtype fun x => Membership.m …
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem H.normalizer x) (H.s …
      hx : Eq (orderOf x) p
      hequiv : Equiv (Subtype fun x => Membership.mem H x) (Subtype fun x => Members …
      y : G
      hy : Membership.mem H y
      ⊢ Membership.mem (H.subgroupOf H.normalizer) ⟨y, ⋯⟩
    -/
    simpa using hy⟩
    /-
      🎉 no goals
    -/


/-- If `H` is a subgroup of `G` of cardinality `p ^ n`,
  then `H` is contained in a subgroup of cardinality `p ^ m`
  if `n ≤ m` and `p ^ m` divides the cardinality of `G` -/
theorem exists_subgroup_card_pow_prime_le [Finite G] (p : ℕ) :
    ∀ {n m : ℕ} [_hp : Fact p.Prime] (_hdvd : p ^ m ∣ Nat.card G) (H : Subgroup G)
      (_hH : Nat.card H = p ^ n) (_hnm : n ≤ m), ∃ K : Subgroup G, Nat.card K = p ^ m ∧ H ≤ K
  | n, m => fun {hdvd H hH hnm} =>
    (lt_or_eq_of_le hnm).elim
      (fun hnm : n < m =>
        have h0m : 0 < m := lt_of_le_of_lt n.zero_le hnm
        have _wf : m - 1 < m := Nat.sub_lt h0m zero_lt_one
        have hnm1 : n ≤ m - 1 := le_tsub_of_add_le_right hnm
        let ⟨K, hK⟩ :=
          @exists_subgroup_card_pow_prime_le _ _ n (m - 1) _
            (Nat.pow_dvd_of_le_of_pow_dvd tsub_le_self hdvd) H hH hnm1
                                                        /-
                                                          G : Type u
                                                          inst✝¹ : Group G
                                                          inst✝ : Finite G
                                                          p : Nat
                                                          _hp✝ : Fact (Nat.Prime p)
                                                          n m : Nat
                                                          hdvd : Dvd.dvd (HPow.hPow p m) (Nat.card G)
                                                          H : Subgroup G
                                                          hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
                                                          hnm✝ : LE.le n m
                                                          hnm : LT.lt n m
                                                          h0m : LT.lt 0 m
                                                          _wf : LT.lt (HSub.hSub m 1) m
                                                          hnm1 : LE.le n (HSub.hSub m 1)
                                                          K : Subgroup G
                                                          hK : And (Eq (Nat.card (Subtype fun x => Membership.mem K x)) (HPow.hPow p (HS …
                                                          ⊢ Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) (Nat.card G)
                                                        -/
        have hdvd' : p ^ (m - 1 + 1) ∣ Nat.card G := by rwa [tsub_add_cancel_of_le h0m.nat_succ_le]
                                                        /-
                                                          🎉 no goals
                                                        -/
        let ⟨K', hK'⟩ := @exists_subgroup_card_pow_succ _ _ _ _ _ _ hdvd' K hK.1
                /-
                  G : Type u
                  inst✝¹ : Group G
                  inst✝ : Finite G
                  p : Nat
                  _hp✝ : Fact (Nat.Prime p)
                  n m : Nat
                  hdvd : Dvd.dvd (HPow.hPow p m) (Nat.card G)
                  H : Subgroup G
                  hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
                  hnm✝ : LE.le n m
                  hnm : LT.lt n m
                  h0m : LT.lt 0 m
                  _wf : LT.lt (HSub.hSub m 1) m
                  hnm1 : LE.le n (HSub.hSub m 1)
                  K : Subgroup G
                  hK : And (Eq (Nat.card (Subtype fun x => Membership.mem K x)) (HPow.hPow p (HS …
                  hdvd' : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HSub.hSub m 1) 1)) (Nat.card G)
                  K' : Subgroup G
                  hK' : And (Eq (Nat.card (Subtype fun x => Membership.mem K' x)) (HPow.hPow p ( …
                  ⊢ Eq (Nat.card (Subtype fun x => Membership.mem K' x)) (HPow.hPow p m)
                -/
        ⟨K', by rw [hK'.1, tsub_add_cancel_of_le h0m.nat_succ_le], le_trans hK.2 hK'.2⟩)
                /-
                  🎉 no goals
                -/
                                /-
                                  G : Type u
                                  inst✝¹ : Group G
                                  inst✝ : Finite G
                                  p : Nat
                                  _hp✝ : Fact (Nat.Prime p)
                                  n m : Nat
                                  hdvd : Dvd.dvd (HPow.hPow p m) (Nat.card G)
                                  H : Subgroup G
                                  hH : Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p n)
                                  hnm✝ : LE.le n m
                                  hnm : Eq n m
                                  ⊢ And (Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hPow p m)) (L …
                                -/
      fun hnm : n = m => ⟨H, by simp [hH, hnm]⟩
                                /-
                                  🎉 no goals
                                -/


/-- A generalisation of **Sylow's first theorem**. If `p ^ n` divides
  the cardinality of `G`, then there is a subgroup of cardinality `p ^ n` -/
theorem exists_subgroup_card_pow_prime [Finite G] (p : ℕ) {n : ℕ} [Fact p.Prime]
    (hdvd : p ^ n ∣ Nat.card G) : ∃ K : Subgroup G, Nat.card K = p ^ n :=
  let ⟨K, hK⟩ := exists_subgroup_card_pow_prime_le p hdvd ⊥
        /-
          G : Type u
          inst✝² : Group G
          inst✝¹ : Finite G
          p n : Nat
          inst✝ : Fact (Nat.Prime p)
          hdvd : Dvd.dvd (HPow.hPow p n) (Nat.card G)
          ⊢ Eq (Nat.card (Subtype fun x => Membership.mem Bot.bot x)) (HPow.hPow p 0)
        -/
    (by rw [card_bot, pow_zero]) n.zero_le
        /-
          🎉 no goals
        -/
  ⟨K, hK.1⟩


/-- A special case of **Sylow's first theorem**. If `G` is a `p`-group of size at least `p ^ n`
then there is a subgroup of cardinality `p ^ n`. -/
lemma exists_subgroup_card_pow_prime_of_le_card {n p : ℕ} (hp : p.Prime) (h : IsPGroup p G)
    (hn : p ^ n ≤ Nat.card G) : ∃ H : Subgroup G, Nat.card H = p ^ n := by
  /-
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    hn : LE.le (HPow.hPow p n) (Nat.card G)
    ⊢ Exists fun H => Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hP …
  -/
  have : Fact p.Prime := ⟨hp⟩
  /-
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    hn : LE.le (HPow.hPow p n) (Nat.card G)
    this : Fact (Nat.Prime p)
    ⊢ Exists fun H => Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hP …
  -/
  have : Finite G := Nat.finite_of_card_ne_zero <| by linarith [Nat.one_le_pow n p hp.pos]
  /-
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    hn : LE.le (HPow.hPow p n) (Nat.card G)
    this✝ : Fact (Nat.Prime p)
    this : Finite G
    ⊢ Exists fun H => Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hP …
  -/
  obtain ⟨m, hm⟩ := h.exists_card_eq
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    hn : LE.le (HPow.hPow p n) (Nat.card G)
    this✝ : Fact (Nat.Prime p)
    this : Finite G
    m : Nat
    hm : Eq (Nat.card G) (HPow.hPow p m)
    ⊢ Exists fun H => Eq (Nat.card (Subtype fun x => Membership.mem H x)) (HPow.hP …
  -/
  refine exists_subgroup_card_pow_prime _ ?_
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    hn : LE.le (HPow.hPow p n) (Nat.card G)
    this✝ : Fact (Nat.Prime p)
    this : Finite G
    m : Nat
    hm : Eq (Nat.card G) (HPow.hPow p m)
    ⊢ Dvd.dvd (HPow.hPow p n) (Nat.card G)
  -/
  rw [hm] at hn ⊢
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    this✝ : Fact (Nat.Prime p)
    this : Finite G
    m : Nat
    hn : LE.le (HPow.hPow p n) (HPow.hPow p m)
    hm : Eq (Nat.card G) (HPow.hPow p m)
    ⊢ Dvd.dvd (HPow.hPow p n) (HPow.hPow p m)
  -/
  exact pow_dvd_pow _ <| (Nat.pow_le_pow_iff_right hp.one_lt).1 hn
  /-
    🎉 no goals
  -/


/-- A special case of **Sylow's first theorem**. If `G` is a `p`-group and `H` a subgroup of size at
least `p ^ n` then there is a subgroup of `H` of cardinality `p ^ n`. -/
lemma exists_subgroup_le_card_pow_prime_of_le_card {n p : ℕ} (hp : p.Prime) (h : IsPGroup p G)
    {H : Subgroup G} (hn : p ^ n ≤ Nat.card H) : ∃ H' ≤ H, Nat.card H' = p ^ n := by
  /-
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hn : LE.le (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem H x))
    ⊢ Exists fun H' => And (LE.le H' H) (Eq (Nat.card (Subtype fun x => Membership …
  -/
  obtain ⟨H', H'card⟩ := exists_subgroup_card_pow_prime_of_le_card hp (h.to_subgroup H) hn
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hn : LE.le (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem H x))
    H' : Subgroup (Subtype fun x => Membership.mem H x)
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p n)
    ⊢ Exists fun H' => And (LE.le H' H) (Eq (Nat.card (Subtype fun x => Membership …
  -/
  refine ⟨H'.map H.subtype, map_subtype_le _, ?_⟩
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hn : LE.le (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem H x))
    H' : Subgroup (Subtype fun x => Membership.mem H x)
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p n)
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.subtype H') x) …
  -/
  rw [← H'card]
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hn : LE.le (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem H x))
    H' : Subgroup (Subtype fun x => Membership.mem H x)
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p n)
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.subtype H') x) …
  -/
  let e : H' ≃* H'.map H.subtype := H'.equivMapOfInjective (Subgroup.subtype H) H.subtype_injective
  /-
    case intro
    G : Type u
    inst✝ : Group G
    n p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hn : LE.le (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem H x))
    H' : Subgroup (Subtype fun x => Membership.mem H x)
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p n)
    e : MulEquiv (Subtype fun x => Membership.mem H' x) (Subtype fun x => Membersh …
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.subtype H') x) …
  -/
  exact Nat.card_congr e.symm.toEquiv
  /-
    🎉 no goals
  -/


/-- A special case of **Sylow's first theorem**. If `G` is a `p`-group and `H` a subgroup of size at
least `k` then there is a subgroup of `H` of cardinality between `k / p` and `k`. -/
lemma exists_subgroup_le_card_le {k p : ℕ} (hp : p.Prime) (h : IsPGroup p G) {H : Subgroup G}
    (hk : k ≤ Nat.card H) (hk₀ : k ≠ 0) : ∃ H' ≤ H, Nat.card H' ≤ k ∧ k < p * Nat.card H' := by
  obtain ⟨m, hmk, hkm⟩ : ∃ s, p ^ s ≤ k ∧ k < p ^ (s + 1) :=
    exists_nat_pow_near (Nat.one_le_iff_ne_zero.2 hk₀) hp.one_lt
  /-
    case intro.intro
    G : Type u
    inst✝ : Group G
    k p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hk : LE.le k (Nat.card (Subtype fun x => Membership.mem H x))
    hk₀ : Ne k 0
    m : Nat
    hmk : LE.le (HPow.hPow p m) k
    hkm : LT.lt k (HPow.hPow p (HAdd.hAdd m 1))
    ⊢ Exists fun H' => And (LE.le H' H) (And (LE.le (Nat.card (Subtype fun x => Me …
  -/
  obtain ⟨H', H'H, H'card⟩ := exists_subgroup_le_card_pow_prime_of_le_card hp h (hmk.trans hk)
  /-
    case intro.intro.intro.intro
    G : Type u
    inst✝ : Group G
    k p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hk : LE.le k (Nat.card (Subtype fun x => Membership.mem H x))
    hk₀ : Ne k 0
    m : Nat
    hmk : LE.le (HPow.hPow p m) k
    hkm : LT.lt k (HPow.hPow p (HAdd.hAdd m 1))
    H' : Subgroup G
    H'H : LE.le H' H
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p m)
    ⊢ Exists fun H' => And (LE.le H' H) (And (LE.le (Nat.card (Subtype fun x => Me …
  -/
  refine ⟨H', H'H, ?_⟩
  /-
    case intro.intro.intro.intro
    G : Type u
    inst✝ : Group G
    k p : Nat
    hp : Nat.Prime p
    h : IsPGroup p G
    H : Subgroup G
    hk : LE.le k (Nat.card (Subtype fun x => Membership.mem H x))
    hk₀ : Ne k 0
    m : Nat
    hmk : LE.le (HPow.hPow p m) k
    hkm : LT.lt k (HPow.hPow p (HAdd.hAdd m 1))
    H' : Subgroup G
    H'H : LE.le H' H
    H'card : Eq (Nat.card (Subtype fun x => Membership.mem H' x)) (HPow.hPow p m)
    ⊢ And (LE.le (Nat.card (Subtype fun x => Membership.mem H' x)) k) (LT.lt k (HM …
  -/
  simpa only [pow_succ', H'card] using And.intro hmk hkm
  /-
    🎉 no goals
  -/


theorem pow_dvd_card_of_pow_dvd_card [Finite G] {p n : ℕ} [hp : Fact p.Prime] (P : Sylow p G)
    (hdvd : p ^ n ∣ Nat.card G) : p ^ n ∣ Nat.card P := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd (HPow.hPow p n) (Nat.card G)
    ⊢ Dvd.dvd (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem (↑P) x))
  -/
  rw [← index_mul_card P.1] at hdvd
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p n : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd (HPow.hPow p n) (HMul.hMul (↑P).index (Nat.card (Subtype fun x  …
    ⊢ Dvd.dvd (HPow.hPow p n) (Nat.card (Subtype fun x => Membership.mem (↑P) x))
  -/
  exact (hp.1.coprime_pow_of_not_dvd P.not_dvd_index).symm.dvd_of_dvd_mul_left hdvd
  /-
    🎉 no goals
  -/


theorem dvd_card_of_dvd_card [Finite G] {p : ℕ} [Fact p.Prime] (P : Sylow p G)
    (hdvd : p ∣ Nat.card G) : p ∣ Nat.card P := by
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd p (Nat.card G)
    ⊢ Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (↑P) x))
  -/
  rw [← pow_one p] at hdvd
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd (HPow.hPow p 1) (Nat.card G)
    ⊢ Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (↑P) x))
  -/
  have key := P.pow_dvd_card_of_pow_dvd_card hdvd
  /-
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd (HPow.hPow p 1) (Nat.card G)
    key : Dvd.dvd (HPow.hPow p 1) (Nat.card (Subtype fun x => Membership.mem (↑P)  …
    ⊢ Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (↑P) x))
  -/
  rwa [pow_one] at key
  /-
    🎉 no goals
  -/


/-- Sylow subgroups are Hall subgroups. -/
theorem card_coprime_index [Finite G] {p : ℕ} [hp : Fact p.Prime] (P : Sylow p G) :
    (Nat.card P).Coprime P.index :=
  let ⟨_n, hn⟩ := IsPGroup.iff_card.mp P.2
  hn.symm ▸ (hp.1.coprime_pow_of_not_dvd P.not_dvd_index).symm


theorem ne_bot_of_dvd_card [Finite G] {p : ℕ} [hp : Fact p.Prime] (P : Sylow p G)
    (hdvd : p ∣ Nat.card G) : (P : Subgroup G) ≠ ⊥ := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd p (Nat.card G)
    ⊢ Ne (↑P) Bot.bot
  -/
  refine fun h => hp.out.not_dvd_one ?_
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd p (Nat.card G)
    h : Eq (↑P) Bot.bot
    ⊢ Dvd.dvd p 1
  -/
  have key : p ∣ Nat.card P := P.dvd_card_of_dvd_card hdvd
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    hdvd : Dvd.dvd p (Nat.card G)
    h : Eq (↑P) Bot.bot
    key : Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (↑P) x))
    ⊢ Dvd.dvd p 1
  -/
  rwa [h, card_bot] at key
  /-
    🎉 no goals
  -/


/-- The cardinality of a Sylow subgroup is `p ^ n`
 where `n` is the multiplicity of `p` in the group order. -/
theorem card_eq_multiplicity [Finite G] {p : ℕ} [hp : Fact p.Prime] (P : Sylow p G) :
    Nat.card P = p ^ Nat.factorization (Nat.card G) p := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p ((Nat.ca …
  -/
  obtain ⟨n, heq : Nat.card P = _⟩ := IsPGroup.iff_card.mp P.isPGroup'
  /-
    case intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    n : Nat
    heq : Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p n)
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p ((Nat.ca …
  -/
  refine Nat.dvd_antisymm ?_ (P.pow_dvd_card_of_pow_dvd_card (Nat.ordProj_dvd _ p))
  /-
    case intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    n : Nat
    heq : Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p n)
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p ((N …
  -/
  rw [heq, ← hp.out.pow_dvd_iff_dvd_ordProj (show Nat.card G ≠ 0 from Nat.card_pos.ne'), ← heq]
  /-
    case intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    hp : Fact (Nat.Prime p)
    P : Sylow p G
    n : Nat
    heq : Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p n)
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (Nat.card G)
  -/
  exact P.1.card_subgroup_dvd_card
  /-
    🎉 no goals
  -/


/-- If `G` has a normal Sylow `p`-subgroup, then it is the only Sylow `p`-subgroup. -/
noncomputable def unique_of_normal {p : ℕ} [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G)
    (h : P.Normal) : Unique (Sylow p G) := by
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    ⊢ Unique (Sylow p G)
  -/
  refine { uniq := fun Q ↦ ?_ }
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    Q : Sylow p G
    ⊢ Eq Q Inhabited.default
  -/
  obtain ⟨x, h1⟩ := exists_smul_eq G P Q
  /-
    case intro
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    Q : Sylow p G
    x : G
    h1 : Eq (HSMul.hSMul x P) Q
    ⊢ Eq Q Inhabited.default
  -/
  obtain ⟨x, h2⟩ := exists_smul_eq G P default
  /-
    case intro.intro
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    Q : Sylow p G
    x✝ : G
    h1 : Eq (HSMul.hSMul x✝ P) Q
    x : G
    h2 : Eq (HSMul.hSMul x P) Inhabited.default
    ⊢ Eq Q Inhabited.default
  -/
  rw [smul_eq_of_normal] at h1 h2
  /-
    case intro.intro
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    Q : Sylow p G
    x✝ : G
    h1 : Eq P Q
    x : G
    h2 : Eq P Inhabited.default
    ⊢ Eq Q Inhabited.default
  -/
  rw [← h1, ← h2]
  /-
    🎉 no goals
  -/


theorem characteristic_of_normal {p : ℕ} [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G)
    (h : P.Normal) : P.Characteristic := by
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    ⊢ (↑P).Characteristic
  -/
  haveI := unique_of_normal P h
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    this : Unique (Sylow p G)
    ⊢ (↑P).Characteristic
  -/
  rw [characteristic_iff_map_eq]
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    this : Unique (Sylow p G)
    ⊢ ∀ (ϕ : MulEquiv G G), Eq (Subgroup.map ϕ.toMonoidHom ↑P) ↑P
  -/
  intro Φ
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    this : Unique (Sylow p G)
    Φ : MulEquiv G G
    ⊢ Eq (Subgroup.map Φ.toMonoidHom ↑P) ↑P
  -/
  show (Φ • P).toSubgroup = P.toSubgroup
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    this : Unique (Sylow p G)
    Φ : MulEquiv G G
    ⊢ Eq ↑(HSMul.hSMul Φ P) ↑P
  -/
  congr
  /-
    case e_self
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    h : (↑P).Normal
    this : Unique (Sylow p G)
    Φ : MulEquiv G G
    ⊢ Eq (HSMul.hSMul Φ P) P
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


theorem normal_of_normalizer_normal {p : ℕ} [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G)
    (hn : P.normalizer.Normal) : P.Normal := by
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    hn : (↑P).normalizer.Normal
    ⊢ (↑P).Normal
  -/
  rw [← normalizer_eq_top_iff, ← normalizer_sup_eq_top' P le_normalizer, sup_idem]
  /-
    🎉 no goals
  -/


@[simp]
theorem normalizer_normalizer {p : ℕ} [Fact p.Prime] [Finite (Sylow p G)] (P : Sylow p G) :
    P.normalizer.normalizer = P.normalizer := by
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    ⊢ Eq (↑P).normalizer.normalizer (↑P).normalizer
  -/
  have := normal_of_normalizer_normal (P.subtype (le_normalizer.trans le_normalizer))
  simp_rw [← normalizer_eq_top_iff, coe_subtype, ← subgroupOf_normalizer_eq le_normalizer, ←
    subgroupOf_normalizer_eq le_rfl, subgroupOf_self] at this
  rw [← range_subtype P.normalizer.normalizer, MonoidHom.range_eq_map,
    ← this trivial]
  /-
    G : Type u
    inst✝² : Group G
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite (Sylow p G)
    P : Sylow p G
    this : True → Eq ((↑P).normalizer.subgroupOf (↑P).normalizer.normalizer) Top.top
    ⊢ Eq (Subgroup.map (↑P).normalizer.normalizer.subtype ((↑P).normalizer.subgrou …
  -/
  exact map_comap_eq_self (le_normalizer.trans (ge_of_eq (range_subtype _)))
  /-
    🎉 no goals
  -/


theorem normal_of_all_max_subgroups_normal [Finite G]
    (hnc : ∀ H : Subgroup G, IsCoatom H → H.Normal) {p : ℕ} [Fact p.Prime] [Finite (Sylow p G)]
    (P : Sylow p G) : P.Normal :=
  normalizer_eq_top_iff.mp
    (by
      /-
        G : Type u
        inst✝³ : Group G
        inst✝² : Finite G
        hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
        p : Nat
        inst✝¹ : Fact (Nat.Prime p)
        inst✝ : Finite (Sylow p G)
        P : Sylow p G
        ⊢ Eq (↑P).normalizer Top.top
      -/
      rcases eq_top_or_exists_le_coatom P.normalizer with (heq | ⟨K, hK, hNK⟩)
        /-
          case inl
          G : Type u
          inst✝³ : Group G
          inst✝² : Finite G
          hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : Finite (Sylow p G)
          P : Sylow p G
          heq : Eq (↑P).normalizer Top.top
          ⊢ Eq (↑P).normalizer Top.top
        -/
      · exact heq
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro
          G : Type u
          inst✝³ : Group G
          inst✝² : Finite G
          hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : Finite (Sylow p G)
          P : Sylow p G
          K : Subgroup G
          hK : IsCoatom K
          hNK : LE.le (↑P).normalizer K
          ⊢ Eq (↑P).normalizer Top.top
        -/
      · haveI := hnc _ hK
        /-
          case inr.intro.intro
          G : Type u
          inst✝³ : Group G
          inst✝² : Finite G
          hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : Finite (Sylow p G)
          P : Sylow p G
          K : Subgroup G
          hK : IsCoatom K
          hNK : LE.le (↑P).normalizer K
          this : K.Normal
          ⊢ Eq (↑P).normalizer Top.top
        -/
        have hPK : P ≤ K := le_trans le_normalizer hNK
        /-
          case inr.intro.intro
          G : Type u
          inst✝³ : Group G
          inst✝² : Finite G
          hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : Finite (Sylow p G)
          P : Sylow p G
          K : Subgroup G
          hK : IsCoatom K
          hNK : LE.le (↑P).normalizer K
          this : K.Normal
          hPK : LE.le (↑P) K
          ⊢ Eq (↑P).normalizer Top.top
        -/
        refine (hK.1 ?_).elim
        /-
          case inr.intro.intro
          G : Type u
          inst✝³ : Group G
          inst✝² : Finite G
          hnc : ∀ (H : Subgroup G), IsCoatom H → H.Normal
          p : Nat
          inst✝¹ : Fact (Nat.Prime p)
          inst✝ : Finite (Sylow p G)
          P : Sylow p G
          K : Subgroup G
          hK : IsCoatom K
          hNK : LE.le (↑P).normalizer K
          this : K.Normal
          hPK : LE.le (↑P) K
          ⊢ Eq K Top.top
        -/
        rw [← sup_of_le_right hNK, P.normalizer_sup_eq_top' hPK])
        /-
          🎉 no goals
        -/


theorem normal_of_normalizerCondition (hnc : NormalizerCondition G) {p : ℕ} [Fact p.Prime]
    [Finite (Sylow p G)] (P : Sylow p G) : P.Normal :=
  normalizer_eq_top_iff.mp <|
    normalizerCondition_iff_only_full_group_self_normalizing.mp hnc _ <| normalizer_normalizer _


/-- If all its Sylow subgroups are normal, then a finite group is isomorphic to the direct product
of these Sylow subgroups.
-/
noncomputable def directProductOfNormal [Finite G]
    (hn : ∀ {p : ℕ} [Fact p.Prime] (P : Sylow p G), P.Normal) :
    (∀ p : (Nat.card G).primeFactors, ∀ P : Sylow p G, P) ≃* G := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem (Nat.card G).primeFactors x)  …
  -/
  have := Fintype.ofFinite G
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this : Fintype G
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem (Nat.card G).primeFactors x)  …
  -/
  set ps := (Nat.card G).primeFactors
  -- “The” Sylow subgroup for p
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem ps x) → (P : Sylow (↑p) G) →  …
  -/
  let P : ∀ p, Sylow p G := default
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    P : (p : Nat) → Sylow p G := Inhabited.default
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem ps x) → (P : Sylow (↑p) G) →  …
  -/
  have : ∀ p, Fintype (P p) := fun p ↦ Fintype.ofFinite (P p)
  have hcomm : Pairwise fun p₁ p₂ : ps => ∀ x y : G, x ∈ P p₁ → y ∈ P p₂ → Commute x y := by
    rintro ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩ hne
    haveI hp₁' := Fact.mk (Nat.prime_of_mem_primeFactors hp₁)
    haveI hp₂' := Fact.mk (Nat.prime_of_mem_primeFactors hp₂)
    have hne' : p₁ ≠ p₂ := by simpa using hne
    apply Subgroup.commute_of_normal_of_disjoint _ _ (hn (P p₁)) (hn (P p₂))
    apply IsPGroup.disjoint_of_ne p₁ p₂ hne' _ _ (P p₁).isPGroup' (P p₂).isPGroup'
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this✝ : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    P : (p : Nat) → Sylow p G := Inhabited.default
    this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
    hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem ps x) → (P : Sylow (↑p) G) →  …
  -/
  refine MulEquiv.trans (N := ∀ p : ps, P p) ?_ ?_
  -- There is only one Sylow subgroup for each p, so the inner product is trivial
  · -- here we need to help the elaborator with an explicit instantiation
    /-
      case refine_1
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ MulEquiv ((p : Subtype fun x => Membership.mem ps x) → (P : Sylow (↑p) G) →  …
    -/
    apply @MulEquiv.piCongrRight ps (fun p => ∀ P : Sylow p G, P) (fun p => P p) _ _
    /-
      case refine_1
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ (j : Subtype fun x => Membership.mem ps x) → MulEquiv ((P : Sylow (↑j) G) →  …
    -/
    rintro ⟨p, hp⟩
    /-
      case refine_1.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p : Nat
      hp : Membership.mem ps p
      ⊢ MulEquiv ((P : Sylow (↑⟨p, hp⟩) G) → Subtype fun x => Membership.mem P x) (S …
    -/
    haveI hp' := Fact.mk (Nat.prime_of_mem_primeFactors hp)
    /-
      case refine_1.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p : Nat
      hp : Membership.mem ps p
      hp' : Fact (Nat.Prime p)
      ⊢ MulEquiv ((P : Sylow (↑⟨p, hp⟩) G) → Subtype fun x => Membership.mem P x) (S …
    -/
    letI := unique_of_normal _ (hn (P p))
    /-
      case refine_1.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝¹ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this✝ : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p : Nat
      hp : Membership.mem ps p
      hp' : Fact (Nat.Prime p)
      this : Unique (Sylow p G) := (P p).unique_of_normal ⋯
      ⊢ MulEquiv ((P : Sylow (↑⟨p, hp⟩) G) → Subtype fun x => Membership.mem P x) (S …
    -/
    apply MulEquiv.piUnique
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this✝ : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    P : (p : Nat) → Sylow p G := Inhabited.default
    this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
    hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
    ⊢ MulEquiv ((p : Subtype fun x => Membership.mem ps x) → Subtype fun x => Memb …
  -/
  apply MulEquiv.ofBijective (Subgroup.noncommPiCoprod hcomm)
  /-
    case refine_2
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this✝ : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    P : (p : Nat) → Sylow p G := Inhabited.default
    this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
    hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
    ⊢ Function.Bijective ⇑(Subgroup.noncommPiCoprod hcomm)
  -/
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  /-
    case refine_2
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
    this✝ : Fintype G
    ps : Finset Nat := (Nat.card G).primeFactors
    P : (p : Nat) → Sylow p G := Inhabited.default
    this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
    hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
    ⊢ And (Function.Injective ⇑(Subgroup.noncommPiCoprod hcomm)) (Eq (Fintype.card …
  -/
  constructor
    /-
      case refine_2.left
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ Function.Injective ⇑(Subgroup.noncommPiCoprod hcomm)
    -/
  · apply Subgroup.injective_noncommPiCoprod_of_iSupIndep
    /-
      case refine_2.left.hind
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ iSupIndep fun i => ↑(P ↑i)
    -/
    apply independent_of_coprime_order hcomm
    /-
      case refine_2.left.hind
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ Pairwise fun i j => (Fintype.card (Subtype fun x => Membership.mem (↑(P ↑i)) …
    -/
    rintro ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩ hne
    /-
      case refine_2.left.hind.mk.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p₁ : Nat
      hp₁ : Membership.mem ps p₁
      p₂ : Nat
      hp₂ : Membership.mem ps p₂
      hne : Ne ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      ⊢ (Fintype.card (Subtype fun x => Membership.mem (↑(P ↑⟨p₁, hp₁⟩)) x)).Coprime …
    -/
    haveI hp₁' := Fact.mk (Nat.prime_of_mem_primeFactors hp₁)
    /-
      case refine_2.left.hind.mk.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p₁ : Nat
      hp₁ : Membership.mem ps p₁
      p₂ : Nat
      hp₂ : Membership.mem ps p₂
      hne : Ne ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      hp₁' : Fact (Nat.Prime p₁)
      ⊢ (Fintype.card (Subtype fun x => Membership.mem (↑(P ↑⟨p₁, hp₁⟩)) x)).Coprime …
    -/
    haveI hp₂' := Fact.mk (Nat.prime_of_mem_primeFactors hp₂)
    /-
      case refine_2.left.hind.mk.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p₁ : Nat
      hp₁ : Membership.mem ps p₁
      p₂ : Nat
      hp₂ : Membership.mem ps p₂
      hne : Ne ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      hp₁' : Fact (Nat.Prime p₁)
      hp₂' : Fact (Nat.Prime p₂)
      ⊢ (Fintype.card (Subtype fun x => Membership.mem (↑(P ↑⟨p₁, hp₁⟩)) x)).Coprime …
    -/
    have hne' : p₁ ≠ p₂ := by simpa using hne
    /-
      case refine_2.left.hind.mk.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p₁ : Nat
      hp₁ : Membership.mem ps p₁
      p₂ : Nat
      hp₂ : Membership.mem ps p₂
      hne : Ne ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      hp₁' : Fact (Nat.Prime p₁)
      hp₂' : Fact (Nat.Prime p₂)
      hne' : Ne p₁ p₂
      ⊢ (Fintype.card (Subtype fun x => Membership.mem (↑(P ↑⟨p₁, hp₁⟩)) x)).Coprime …
    -/
    simp only [← Nat.card_eq_fintype_card]
    /-
      case refine_2.left.hind.mk.mk
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      p₁ : Nat
      hp₁ : Membership.mem ps p₁
      p₂ : Nat
      hp₂ : Membership.mem ps p₂
      hne : Ne ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      hp₁' : Fact (Nat.Prime p₁)
      hp₂' : Fact (Nat.Prime p₂)
      hne' : Ne p₁ p₂
      ⊢ (Nat.card (Subtype fun x => Membership.mem (↑(P p₁)) x)).Coprime (Nat.card ( …
    -/
    apply IsPGroup.coprime_card_of_ne p₁ p₂ hne' _ _ (P p₁).isPGroup' (P p₂).isPGroup'
    /-
      🎉 no goals
    -/
    /-
      case refine_2.right
      G : Type u
      inst✝¹ : Group G
      inst✝ : Finite G
      hn : ∀ {p : Nat} [inst : Fact (Nat.Prime p)] (P : Sylow p G), (↑P).Normal
      this✝ : Fintype G
      ps : Finset Nat := (Nat.card G).primeFactors
      P : (p : Nat) → Sylow p G := Inhabited.default
      this : (p : Nat) → Fintype (Subtype fun x => Membership.mem (↑(P p)) x)
      hcomm : Pairwise fun p₁ p₂ => ∀ (x y : G), Membership.mem (P ↑p₁) x → Membersh …
      ⊢ Eq (Fintype.card ((i : Subtype fun x => Membership.mem ps x) → Subtype fun x …
    -/
  · simp only [← Nat.card_eq_fintype_card]
    calc
      Nat.card (∀ p : ps, P p) = ∏ p : ps, Nat.card (P p) := Nat.card_pi
      _ = ∏ p : ps, p.1 ^ (Nat.card G).factorization p.1 := by
        congr 1 with ⟨p, hp⟩
        exact @card_eq_multiplicity _ _ _ p ⟨Nat.prime_of_mem_primeFactors hp⟩ (P p)
      _ = ∏ p ∈ ps, p ^ (Nat.card G).factorization p :=
        (Finset.prod_finset_coe (fun p => p ^ (Nat.card G).factorization p) _)
      _ = (Nat.card G).factorization.prod (· ^ ·) := rfl
      _ = Nat.card G := Nat.factorization_prod_pow_eq_self Nat.card_pos.ne'



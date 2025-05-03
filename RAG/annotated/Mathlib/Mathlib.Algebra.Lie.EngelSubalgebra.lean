/-- The Engel subalgebra `Engel R x` consists of
all `y : L` such that `(ad R L x)^n` kills `y` for some `n`.

Engel subalgebras are self-normalizing (`LieSubalgebra.normalizer_engel`),
and minimal ones are nilpotent, hence Cartan subalgebras. -/
@[simps!]
def engel (x : L) : LieSubalgebra R L :=
  { (ad R L x).maxGenEigenspace 0 with
    lie_mem' := by
      simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
        Submodule.mem_toAddSubmonoid, Module.End.mem_maxGenEigenspace, zero_smul,
        sub_zero, forall_exists_index]
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x : L
        ⊢ ∀ {x_1 y : L} (x_2 : Nat), Eq ((HPow.hPow ((LieAlgebra.ad R L) x) x_2) x_1)  …
      -/
      intro y z m hm n hn
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ⊢ Exists fun k => Eq ((HPow.hPow ((LieAlgebra.ad R L) x) k) (Bracket.bracket y …
      -/
      refine ⟨m + n, ?_⟩
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) x) (HAdd.hAdd m n)) (Bracket.bracket y z …
      -/
      rw [ad_pow_lie]
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd m n)).sum fun ij => HSMu …
      -/
      apply Finset.sum_eq_zero
      /-
        case h
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ⊢ ∀ (x_1 : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal  …
      -/
      intro ij hij
      /-
        case h
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ij : Prod Nat Nat
        hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd m n)) ij
        ⊢ Eq (HSMul.hSMul ((HAdd.hAdd m n).choose ij.1) (Bracket.bracket ((HPow.hPow ( …
      -/
      obtain (h|h) : m ≤ ij.1 ∨ n ≤ ij.2 := by rw [Finset.mem_antidiagonal] at hij; omega
      /-
        case h.inl
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        x y z : L
        m : Nat
        hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) m) y) 0
        n : Nat
        hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) z) 0
        ij : Prod Nat Nat
        hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd m n)) ij
        h : LE.le m ij.1
        ⊢ Eq (HSMul.hSMul ((HAdd.hAdd m n).choose ij.1) (Bracket.bracket ((HPow.hPow ( …
      -/
      all_goals simp [LinearMap.pow_map_zero_of_le h, hm, hn] }
      /-
        🎉 no goals
      -/


lemma mem_engel_iff (x y : L) :
    y ∈ engel R x ↔ ∃ n : ℕ, ((ad R L x) ^ n) y = 0 :=
                                                      /-
                                                        R : Type u_1
                                                        L : Type u_2
                                                        inst✝² : CommRing R
                                                        inst✝¹ : LieRing L
                                                        inst✝ : LieAlgebra R L
                                                        x y : L
                                                        ⊢ Iff (Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieAlgebra.ad R L) x) (HSMu …
                                                      -/
  (Module.End.mem_maxGenEigenspace _ _ _).trans <| by simp only [zero_smul, sub_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma self_mem_engel (x : L) : x ∈ engel R x := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Membership.mem (LieSubalgebra.engel R x) x
  -/
  simp only [mem_engel_iff]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Exists fun n => Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) x) 0
  -/
  exact ⟨1, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma engel_zero : engel R (0 : L) = ⊤ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Eq (LieSubalgebra.engel R 0) Top.top
  -/
  rw [eq_top_iff]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ LE.le Top.top (LieSubalgebra.engel R 0)
  -/
  rintro x -
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Membership.mem (LieSubalgebra.engel R 0) x
  -/
  rw [mem_engel_iff, LieHom.map_zero]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Exists fun n => Eq ((HPow.hPow 0 n) x) 0
  -/
  use 1
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Eq ((HPow.hPow 0 1) x) 0
  -/
  simp only [pow_one, LinearMap.zero_apply]
  /-
    🎉 no goals
  -/


/-- Engel subalgebras are self-normalizing.
See `LieSubalgebra.normalizer_eq_self_of_engel_le` for a proof that Lie-subalgebras
containing an Engel subalgebra are also self-normalizing,
provided that the ambient Lie algebra is artinina. -/
@[simp]
lemma normalizer_engel (x : L) : normalizer (engel R x) = engel R x := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Eq (LieSubalgebra.engel R x).normalizer (LieSubalgebra.engel R x)
  -/
  apply le_antisymm _ (le_normalizer _)
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ LE.le (LieSubalgebra.engel R x).normalizer (LieSubalgebra.engel R x)
  -/
  intro y hy
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    hy : Membership.mem (LieSubalgebra.engel R x).normalizer y
    ⊢ Membership.mem (LieSubalgebra.engel R x) y
  -/
  rw [mem_normalizer_iff] at hy
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    hy : ∀ (y_1 : L), Membership.mem (LieSubalgebra.engel R x) y_1 → Membership.me …
    ⊢ Membership.mem (LieSubalgebra.engel R x) y
  -/
  specialize hy x (self_mem_engel R x)
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    hy : Membership.mem (LieSubalgebra.engel R x) (Bracket.bracket y x)
    ⊢ Membership.mem (LieSubalgebra.engel R x) y
  -/
  rw [← lie_skew, neg_mem_iff (G := L), mem_engel_iff] at hy
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    hy : Exists fun n => Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (Bracket.bracke …
    ⊢ Membership.mem (LieSubalgebra.engel R x) y
  -/
  rcases hy with ⟨n, hn⟩
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    n : Nat
    hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (Bracket.bracket x y)) 0
    ⊢ Membership.mem (LieSubalgebra.engel R x) y
  -/
  rw [mem_engel_iff]
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    n : Nat
    hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (Bracket.bracket x y)) 0
    ⊢ Exists fun n => Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) y) 0
  -/
  use n+1
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    n : Nat
    hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (Bracket.bracket x y)) 0
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) x) (HAdd.hAdd n 1)) y) 0
  -/
  rw [pow_succ, LinearMap.mul_apply]
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x y : L
    n : Nat
    hn : Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (Bracket.bracket x y)) 0
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (((LieAlgebra.ad R L) x) y)) 0
  -/
  exact hn
  /-
    🎉 no goals
  -/


open Filter in
/-- A Lie-subalgebra of an Artinian Lie algebra is self-normalizing
if it contains an Engel subalgebra.
See `LieSubalgebra.normalizer_engel` for a proof that Engel subalgebras are self-normalizing,
avoiding the Artinian condition. -/
lemma normalizer_eq_self_of_engel_le [IsArtinian R L]
    (H : LieSubalgebra R L) (x : L) (h : engel R x ≤ H) :
    normalizer H = H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    ⊢ Eq H.normalizer H
  -/
  set N := normalizer H
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    ⊢ Eq N H
  -/
  apply le_antisymm _ (le_normalizer H)
  calc N.toSubmodule ≤ (engel R x).toSubmodule ⊔ H.toSubmodule := ?_
       _ = H := by rwa [sup_eq_right]
  have aux₁ : ∀ n ∈ N, ⁅x, n⁆ ∈ H := by
    intro n hn
    rw [mem_normalizer_iff] at hn
    specialize hn x (h (self_mem_engel R x))
    rwa [← lie_skew, neg_mem_iff (G := L)]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
    ⊢ LE.le N.toSubmodule (Max.max (LieSubalgebra.engel R x).toSubmodule H.toSubmo …
  -/
  have aux₂ : ∀ n ∈ N, ⁅x, n⁆ ∈ N := fun n hn ↦ le_normalizer H (aux₁ _ hn)
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
    aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
    ⊢ LE.le N.toSubmodule (Max.max (LieSubalgebra.engel R x).toSubmodule H.toSubmo …
  -/
  let dx : N →ₗ[R] N := (ad R L x).restrict aux₂
  obtain ⟨k, hk⟩ : ∃ a, ∀ b ≥ a, Codisjoint (LinearMap.ker (dx ^ b)) (LinearMap.range (dx ^ b)) :=
    eventually_atTop.mp <| dx.eventually_codisjoint_ker_pow_range_pow
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
    aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
    dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → Codisjoint (LinearMap.ker (HPow.hPow dx b)) (Lin …
    ⊢ LE.le N.toSubmodule (Max.max (LieSubalgebra.engel R x).toSubmodule H.toSubmo …
  -/
  specialize hk (k+1) (Nat.le_add_right k 1)
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
    aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
    dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
    k : Nat
    hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
    ⊢ LE.le N.toSubmodule (Max.max (LieSubalgebra.engel R x).toSubmodule H.toSubmo …
  -/
  rw [← Submodule.map_subtype_top N.toSubmodule, Submodule.map_le_iff_le_comap]
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsArtinian R L
    H : LieSubalgebra R L
    x : L
    h : LE.le (LieSubalgebra.engel R x) H
    N : LieSubalgebra R L := H.normalizer
    aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
    aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
    dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
    k : Nat
    hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
    ⊢ LE.le Top.top (Submodule.comap N.subtype (Max.max (LieSubalgebra.engel R x). …
  -/
  apply hk
    /-
      case intro.a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (Submodule.comap N.subt …
    -/
  · rw [← Submodule.map_le_iff_le_comap]
    /-
      case intro.a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (Submodule.map N.subtype (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1)) …
    -/
    apply le_sup_of_le_left
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (Submodule.map N.subtype (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1)) …
    -/
    rw [Submodule.map_le_iff_le_comap]
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (Submodule.comap N.subt …
    -/
    intro y hy
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      hy : Membership.mem (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) y
      ⊢ Membership.mem (Submodule.comap N.subtype (LieSubalgebra.engel R x).toSubmod …
    -/
    simp only [Submodule.mem_comap, mem_engel_iff, mem_toSubmodule]
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      hy : Membership.mem (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) y
      ⊢ Exists fun n => Eq ((HPow.hPow ((LieAlgebra.ad R L) x) n) (N.subtype y)) 0
    -/
    use k+1
    /-
      case h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      hy : Membership.mem (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) y
      ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) x) (HAdd.hAdd k 1)) (N.subtype y)) 0
    -/
    clear hk; revert hy
    /-
      case h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      y : Subtype fun x => Membership.mem N x
      ⊢ Membership.mem (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) y → Eq ((HPow. …
    -/
    generalize k+1 = k
    induction k generalizing y with
    | zero =>
      cases y; intro hy; simp only [pow_zero, LinearMap.one_apply]
      exact (AddSubmonoid.mk_eq_zero N.toAddSubmonoid).mp hy
    | succ k ih => simp only [pow_succ, LinearMap.mem_ker, LinearMap.mul_apply] at ih ⊢; apply ih
    /-
      case intro.a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (LinearMap.range (HPow.hPow dx (HAdd.hAdd k 1))) (Submodule.comap N.su …
    -/
  · rw [← Submodule.map_le_iff_le_comap]
    /-
      case intro.a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (Submodule.map N.subtype (LinearMap.range (HPow.hPow dx (HAdd.hAdd k 1 …
    -/
    apply le_sup_of_le_right
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (Submodule.map N.subtype (LinearMap.range (HPow.hPow dx (HAdd.hAdd k 1 …
    -/
    rw [Submodule.map_le_iff_le_comap]
    /-
      case intro.a.h
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      ⊢ LE.le (LinearMap.range (HPow.hPow dx (HAdd.hAdd k 1))) (Submodule.comap N.su …
    -/
    rintro _ ⟨y, rfl⟩
    /-
      case intro.a.h.intro
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      ⊢ Membership.mem (Submodule.comap N.subtype H.toSubmodule) ((HPow.hPow dx (HAd …
    -/
    simp only [pow_succ', LinearMap.mul_apply, Submodule.mem_comap, mem_toSubmodule]
    /-
      case intro.a.h.intro
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      ⊢ Membership.mem H (N.subtype (dx ((HPow.hPow dx k) y)))
    -/
    apply aux₁
    /-
      case intro.a.h.intro.a
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsArtinian R L
      H : LieSubalgebra R L
      x : L
      h : LE.le (LieSubalgebra.engel R x) H
      N : LieSubalgebra R L := H.normalizer
      aux₁ : ∀ (n : L), Membership.mem N n → Membership.mem H (Bracket.bracket x n)
      aux₂ : ∀ (n : L), Membership.mem N n → Membership.mem N (Bracket.bracket x n)
      dx : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) (Subtype f …
      k : Nat
      hk : Codisjoint (LinearMap.ker (HPow.hPow dx (HAdd.hAdd k 1))) (LinearMap.rang …
      y : Subtype fun x => Membership.mem N x
      ⊢ Membership.mem N (N.subtype ((HPow.hPow dx k) y))
    -/
    simp only [Submodule.coe_subtype, SetLike.coe_mem]
    /-
      🎉 no goals
    -/


/-- A Lie subalgebra of a Noetherian Lie algebra is nilpotent
if it is contained in the Engel subalgebra of all its elements. -/
lemma isNilpotent_of_forall_le_engel [IsNoetherian R L]
    (H : LieSubalgebra R L) (h : ∀ x ∈ H, H ≤ engel R x) :
    LieAlgebra.IsNilpotent R H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    ⊢ LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
  -/
  rw [LieAlgebra.isNilpotent_iff_forall]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    ⊢ ∀ (x : Subtype fun x => Membership.mem H x), IsNilpotent ((LieAlgebra.ad R ( …
  -/
  intro x
  let K : ℕ →o Submodule R H :=
    ⟨fun n ↦ LinearMap.ker ((ad R H x) ^ n), fun m n hmn ↦ ?mono⟩
  case mono =>
    intro y hy
    rw [LinearMap.mem_ker] at hy ⊢
    exact LinearMap.pow_map_zero_of_le hmn hy
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    ⊢ IsNilpotent ((LieAlgebra.ad R (Subtype fun x => Membership.mem H x)) x)
  -/
  obtain ⟨n, hn⟩ := monotone_stabilizes_iff_noetherian.mpr inferInstance K
  /-
    case intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    ⊢ IsNilpotent ((LieAlgebra.ad R (Subtype fun x => Membership.mem H x)) x)
  -/
  use n
  /-
    case h
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    ⊢ Eq (HPow.hPow ((LieAlgebra.ad R (Subtype fun x => Membership.mem H x)) x) n) 0
  -/
  ext y
  /-
    case h.h.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    y : Subtype fun x => Membership.mem H x
    ⊢ Eq ↑((HPow.hPow ((LieAlgebra.ad R (Subtype fun x => Membership.mem H x)) x)  …
  -/
  rw [coe_ad_pow]
  /-
    case h.h.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    h : ∀ (x : L), Membership.mem H x → LE.le H (LieSubalgebra.engel R x)
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    y : Subtype fun x => Membership.mem H x
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
  -/
  specialize h x x.2 y.2
  /-
    case h.h.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    y : Subtype fun x => Membership.mem H x
    h : Membership.mem (LieSubalgebra.engel R ↑x) ↑y
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
  -/
  rw [mem_engel_iff] at h
  /-
    case h.h.a
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    y : Subtype fun x => Membership.mem H x
    h : Exists fun n => Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) 0
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
  -/
  obtain ⟨m, hm⟩ := h
  /-
    case h.h.a.intro
    R : Type u_1
    L : Type u_2
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    H : LieSubalgebra R L
    x : Subtype fun x => Membership.mem H x
    K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
    y : Subtype fun x => Membership.mem H x
    m : Nat
    hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) m) ↑y) 0
    ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
  -/
  obtain (hmn|hmn) : m ≤ n ∨ n ≤ m := le_total m n
    /-
      case h.h.a.intro.inl
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      H : LieSubalgebra R L
      x : Subtype fun x => Membership.mem H x
      K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
      y : Subtype fun x => Membership.mem H x
      m : Nat
      hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) m) ↑y) 0
      hmn : LE.le m n
      ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
    -/
  · exact LinearMap.pow_map_zero_of_le hmn hm
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.intro.inr
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      H : LieSubalgebra R L
      x : Subtype fun x => Membership.mem H x
      K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
      y : Subtype fun x => Membership.mem H x
      m : Nat
      hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) m) ↑y) 0
      hmn : LE.le n m
      ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
    -/
  · have : ∀ k : ℕ, ((ad R L) x ^ k) y = 0 ↔ y ∈ K k := by simp [K, Subtype.ext_iff, coe_ad_pow]
    /-
      case h.h.a.intro.inr
      R : Type u_1
      L : Type u_2
      inst✝³ : CommRing R
      inst✝² : LieRing L
      inst✝¹ : LieAlgebra R L
      inst✝ : IsNoetherian R L
      H : LieSubalgebra R L
      x : Subtype fun x => Membership.mem H x
      K : OrderHom Nat (Submodule R (Subtype fun x => Membership.mem H x)) := { toFu …
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (K n) (K m)
      y : Subtype fun x => Membership.mem H x
      m : Nat
      hm : Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) m) ↑y) 0
      hmn : LE.le n m
      this : ∀ (k : Nat), Iff (Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) k) ↑y) 0) (Me …
      ⊢ Eq ((HPow.hPow ((LieAlgebra.ad R L) ↑x) n) ↑y) ↑(0 y)
    -/
    rwa [this, ← hn m hmn, ← this] at hm
    /-
      🎉 no goals
    -/



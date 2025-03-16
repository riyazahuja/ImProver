theorem Submodule.mem_of_localization_maximal (m : M) (N : Submodule R M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], f P m ∈ N.localized₀ P.primeCompl (f P)) :
    m ∈ N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    ⊢ Membership.mem N m
  -/
  let I : Ideal R := N.comap (LinearMap.toSpanSingleton R M m)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ⊢ Membership.mem N m
  -/
  suffices I = ⊤ by simpa [I] using I.eq_top_iff_one.mp this
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ⊢ Eq I Top.top
  -/
  refine Not.imp_symm I.exists_le_maximal fun ⟨P, hP, le⟩ ↦ ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    x✝ : Exists fun M => And M.IsMaximal (LE.le I M)
    P : Ideal R
    hP : P.IsMaximal
    le : LE.le I P
    ⊢ False
  -/
  obtain ⟨a, ha, s, e⟩ := h P
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    x✝ : Exists fun M => And M.IsMaximal (LE.le I M)
    P : Ideal R
    hP : P.IsMaximal
    le : LE.le I P
    a : M
    ha : Membership.mem N a
    s : Subtype fun x => Membership.mem P.primeCompl x
    e : Eq (IsLocalizedModule.mk' (f P) a s) ((f P) m)
    ⊢ False
  -/
  rw [← IsLocalizedModule.mk'_one P.primeCompl, IsLocalizedModule.mk'_eq_mk'_iff] at e
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    x✝ : Exists fun M => And M.IsMaximal (LE.le I M)
    P : Ideal R
    hP : P.IsMaximal
    le : LE.le I P
    a : M
    ha : Membership.mem N a
    s : Subtype fun x => Membership.mem P.primeCompl x
    e : Exists fun s_1 => Eq (HSMul.hSMul s_1 (HSMul.hSMul s m)) (HSMul.hSMul s_1  …
    ⊢ False
  -/
  obtain ⟨t, ht⟩ := e
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    x✝ : Exists fun M => And M.IsMaximal (LE.le I M)
    P : Ideal R
    hP : P.IsMaximal
    le : LE.le I P
    a : M
    ha : Membership.mem N a
    s t : Subtype fun x => Membership.mem P.primeCompl x
    ht : Eq (HSMul.hSMul t (HSMul.hSMul s m)) (HSMul.hSMul t (HSMul.hSMul 1 a))
    ⊢ False
  -/
  simp_rw [smul_smul] at ht
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m : M
    N : Submodule R M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Membership.mem (Submodule.localized₀ …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    x✝ : Exists fun M => And M.IsMaximal (LE.le I M)
    P : Ideal R
    hP : P.IsMaximal
    le : LE.le I P
    a : M
    ha : Membership.mem N a
    s t : Subtype fun x => Membership.mem P.primeCompl x
    ht : Eq (HSMul.hSMul (HMul.hMul t s) m) (HSMul.hSMul (HMul.hMul t 1) a)
    ⊢ False
  -/
  exact (t * s).2 (le <| by apply ht ▸ smul_mem _ _ ha)
  /-
    🎉 no goals
  -/


/-- Let `N₁ N₂ : Submodule R M`. If the localization of `N₁` at each maximal ideal `P` is
included in the localization of `N₂` at `P`, then `N₁ ≤ N₂`. -/
theorem Submodule.le_of_localization_maximal {N₁ N₂ : Submodule R M}
    (h : ∀ (P : Ideal R) [P.IsMaximal],
      N₁.localized₀ P.primeCompl (f P) ≤ N₂.localized₀ P.primeCompl (f P)) :
    N₁ ≤ N₂ :=
                                                                              /-
                                                                                R : Type u_1
                                                                                M : Type u_2
                                                                                inst✝⁵ : CommSemiring R
                                                                                inst✝⁴ : AddCommMonoid M
                                                                                inst✝³ : Module R M
                                                                                Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                                                inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                                                inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                                                f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                                                inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                                                N₁ N₂ : Submodule R M
                                                                                h : ∀ (P : Ideal R) [inst : P.IsMaximal], LE.le (Submodule.localized₀ P.primeC …
                                                                                m : M
                                                                                hm : Membership.mem N₁ m
                                                                                P : Ideal R
                                                                                hP : P.IsMaximal
                                                                                ⊢ Eq (IsLocalizedModule.mk' (f P) m 1) ((f P) m)
                                                                              -/
  fun m hm ↦ mem_of_localization_maximal _ f _ _ fun P hP ↦ h P ⟨m, hm, 1, by simp⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- Let `N₁ N₂ : Submodule R M`. If the localization of `N₁` at each maximal ideal `P` is equal to
the localization of `N₂` at `P`, then `N₁ = N₂`. -/
theorem Submodule.eq_of_localization₀_maximal {N₁ N₂ : Submodule R M}
    (h : ∀ (P : Ideal R) [P.IsMaximal],
      N₁.localized₀ P.primeCompl (f P) = N₂.localized₀ P.primeCompl (f P)) :
    N₁ = N₂ :=
  le_antisymm (Submodule.le_of_localization_maximal Mₚ f fun P _ ↦ (h P).le)
    (Submodule.le_of_localization_maximal Mₚ f fun P _ ↦ (h P).ge)


/-- A submodule is trivial if its localization at every maximal ideal is trivial. -/
theorem Submodule.eq_bot_of_localization₀_maximal (N : Submodule R M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], N.localized₀ P.primeCompl (f P) = ⊥) :
    N = ⊥ :=
                                                           /-
                                                             R : Type u_1
                                                             M : Type u_2
                                                             inst✝⁵ : CommSemiring R
                                                             inst✝⁴ : AddCommMonoid M
                                                             inst✝³ : Module R M
                                                             Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                             inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                             inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                             f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                             inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                             N : Submodule R M
                                                             h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq (Submodule.localized₀ P.primeComp …
                                                             P : Ideal R
                                                             hP : P.IsMaximal
                                                             ⊢ Eq (Submodule.localized₀ P.primeCompl (f P) N) (Submodule.localized₀ P.prime …
                                                           -/
  Submodule.eq_of_localization₀_maximal Mₚ f fun P hP ↦ by simpa using h P
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem Submodule.eq_top_of_localization₀_maximal (N : Submodule R M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], N.localized₀ P.primeCompl (f P) = ⊤) :
    N = ⊤ :=
                                                           /-
                                                             R : Type u_1
                                                             M : Type u_2
                                                             inst✝⁵ : CommSemiring R
                                                             inst✝⁴ : AddCommMonoid M
                                                             inst✝³ : Module R M
                                                             Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                             inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                             inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                             f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                             inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                             N : Submodule R M
                                                             h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq (Submodule.localized₀ P.primeComp …
                                                             P : Ideal R
                                                             hP : P.IsMaximal
                                                             ⊢ Eq (Submodule.localized₀ P.primeCompl (f P) N) (Submodule.localized₀ P.prime …
                                                           -/
  Submodule.eq_of_localization₀_maximal Mₚ f fun P hP ↦ by simpa using h P
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem Module.eq_of_localization_maximal (m m' : M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], f P m = f P m') :
    m = m' := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m m' : M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) ((f P) m')
    ⊢ Eq m m'
  -/
  rw [← one_smul R m, ← one_smul R m']
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m m' : M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) ((f P) m')
    ⊢ Eq (HSMul.hSMul 1 m) (HSMul.hSMul 1 m')
  -/
  by_contra ne
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m m' : M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) ((f P) m')
    ne : Not (Eq (HSMul.hSMul 1 m) (HSMul.hSMul 1 m'))
    ⊢ False
  -/
  have ⟨P, mP, le⟩ := (eqIdeal R m m').exists_le_maximal ((Ideal.ne_top_iff_one _).mpr ne)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m m' : M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) ((f P) m')
    ne : Not (Eq (HSMul.hSMul 1 m) (HSMul.hSMul 1 m'))
    P : Ideal R
    mP : P.IsMaximal
    le : LE.le (Module.eqIdeal R m m') P
    ⊢ False
  -/
  have ⟨s, hs⟩ := (IsLocalizedModule.eq_iff_exists P.primeCompl _).mp (h P)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    m m' : M
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) ((f P) m')
    ne : Not (Eq (HSMul.hSMul 1 m) (HSMul.hSMul 1 m'))
    P : Ideal R
    mP : P.IsMaximal
    le : LE.le (Module.eqIdeal R m m') P
    s : Subtype fun x => Membership.mem P.primeCompl x
    hs : Eq (HSMul.hSMul s m) (HSMul.hSMul s m')
    ⊢ False
  -/
  exact s.2 (le hs)
  /-
    🎉 no goals
  -/


theorem Module.eq_zero_of_localization_maximal (m : M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], f P m = 0) :
    m = 0 :=
                                                  /-
                                                    R : Type u_1
                                                    M : Type u_2
                                                    inst✝⁵ : CommSemiring R
                                                    inst✝⁴ : AddCommMonoid M
                                                    inst✝³ : Module R M
                                                    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                    m : M
                                                    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((f P) m) 0
                                                    P : Ideal R
                                                    x✝ : P.IsMaximal
                                                    ⊢ Eq ((f P) m) ((f P) 0)
                                                  -/
  eq_of_localization_maximal _ f _ _ fun P _ ↦ by rw [h, map_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem LinearMap.eq_of_localization_maximal (g g' : M →ₗ[R] M₁)
    (h : ∀ (P : Ideal R) [P.IsMaximal],
      IsLocalizedModule.map P.primeCompl (f P) (f₁ P) g =
      IsLocalizedModule.map P.primeCompl (f P) (f₁ P) g') :
    g = g' :=
  ext fun x ↦ Module.eq_of_localization_maximal _ f₁ _ _ fun P _ ↦ by
    /-
      R : Type u_1
      M : Type u_2
      M₁ : Type u_3
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : Module R M₁
      Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
      inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
      inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
      f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
      inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
      M₁ₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
      inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (M₁ₚ P)
      inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (M₁ₚ P)
      f₁ : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M₁ (M₁ₚ P)
      inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
      g g' : LinearMap (RingHom.id R) M M₁
      h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq ((IsLocalizedModule.map P.primeCo …
      x : M
      P : Ideal R
      x✝ : P.IsMaximal
      ⊢ Eq ((f₁ P) (g x)) ((f₁ P) (g' x))
    -/
    simpa only [IsLocalizedModule.map_apply] using DFunLike.congr_fun (h P) (f P x)
    /-
      🎉 no goals
    -/


include f in
theorem Module.subsingleton_of_localization_maximal
    (h : ∀ (P : Ideal R) [P.IsMaximal], Subsingleton (Mₚ P)) :
    Subsingleton M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Subsingleton (Mₚ P)
    ⊢ Subsingleton M
  -/
  rw [subsingleton_iff_forall_eq 0]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Subsingleton (Mₚ P)
    ⊢ ∀ (y : M), Eq y 0
  -/
  intro x
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    h : ∀ (P : Ideal R) [inst : P.IsMaximal], Subsingleton (Mₚ P)
    x : M
    ⊢ Eq x 0
  -/
  exact Module.eq_of_localization_maximal Mₚ f x 0 fun _ _ ↦ Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


theorem Submodule.eq_of_localization_maximal {N₁ N₂ : Submodule R M}
    (h : ∀ (P : Ideal R) [P.IsMaximal],
      N₁.localized' (Rₚ P) P.primeCompl (f P) = N₂.localized' (Rₚ P) P.primeCompl (f P)) :
    N₁ = N₂ :=
  eq_of_localization₀_maximal Mₚ f fun P _ ↦ congr(restrictScalars _ $(h P))


theorem Submodule.eq_bot_of_localization_maximal (N : Submodule R M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], N.localized' (Rₚ P) P.primeCompl (f P) = ⊥) :
    N = ⊥ :=
                                                             /-
                                                               R : Type u_1
                                                               M : Type u_2
                                                               inst✝¹⁰ : CommSemiring R
                                                               inst✝⁹ : AddCommMonoid M
                                                               inst✝⁸ : Module R M
                                                               Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
                                                               inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommSemiring (Rₚ P)
                                                               inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
                                                               inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
                                                               Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                               inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                               inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                               inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
                                                               inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
                                                               f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                               inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                               N : Submodule R M
                                                               h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq (Submodule.localized' (Rₚ P) P.pr …
                                                               P : Ideal R
                                                               hP : P.IsMaximal
                                                               ⊢ Eq (Submodule.localized' (Rₚ P) P.primeCompl (f P) N) (Submodule.localized'  …
                                                             -/
  Submodule.eq_of_localization_maximal Rₚ Mₚ f fun P hP ↦ by simpa using h P
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Submodule.eq_top_of_localization_maximal (N : Submodule R M)
    (h : ∀ (P : Ideal R) [P.IsMaximal], N.localized' (Rₚ P) P.primeCompl (f P) = ⊤) :
    N = ⊤ :=
                                                             /-
                                                               R : Type u_1
                                                               M : Type u_2
                                                               inst✝¹⁰ : CommSemiring R
                                                               inst✝⁹ : AddCommMonoid M
                                                               inst✝⁸ : Module R M
                                                               Rₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_4
                                                               inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → CommSemiring (Rₚ P)
                                                               inst✝⁶ : (P : Ideal R) → [inst : P.IsMaximal] → Algebra R (Rₚ P)
                                                               inst✝⁵ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalization.AtPrime (Rₚ P) P
                                                               Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                               inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                               inst✝³ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                               inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → Module (Rₚ P) (Mₚ P)
                                                               inst✝¹ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsScalarTower R (Rₚ P) (Mₚ P)
                                                               f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                               inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                               N : Submodule R M
                                                               h : ∀ (P : Ideal R) [inst : P.IsMaximal], Eq (Submodule.localized' (Rₚ P) P.pr …
                                                               P : Ideal R
                                                               hP : P.IsMaximal
                                                               ⊢ Eq (Submodule.localized' (Rₚ P) P.primeCompl (f P) N) (Submodule.localized'  …
                                                             -/
  Submodule.eq_of_localization_maximal Rₚ Mₚ f fun P hP ↦ by simpa using h P
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Module.eq_of_isLocalized_span (x y : M) (h : ∀ r : s, f r x = f r y) : x = y := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ⊢ Eq x y
  -/
  suffices Module.eqIdeal R x y = ⊤ by simpa [Module.eqIdeal] using (eq_top_iff_one _).mp this
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ⊢ Eq (Module.eqIdeal R x y) Top.top
  -/
  by_contra ne
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ne : Not (Eq (Module.eqIdeal R x y) Top.top)
    ⊢ False
  -/
  have ⟨r, hrs, disj⟩ := exists_disjoint_powers_of_span_eq_top s span_eq _ ne
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ne : Not (Eq (Module.eqIdeal R x y) Top.top)
    r : R
    hrs : Membership.mem s r
    disj : Disjoint ↑(Module.eqIdeal R x y) ↑(Submonoid.powers r)
    ⊢ False
  -/
  let r : s := ⟨r, hrs⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ne : Not (Eq (Module.eqIdeal R x y) Top.top)
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑(Module.eqIdeal R x y) ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    ⊢ False
  -/
  have ⟨⟨_, n, rfl⟩, eq⟩ := (IsLocalizedModule.eq_iff_exists (.powers r.1) _).mp (h r)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    x y : M
    h : ∀ (r : ↑s), Eq ((f r) x) ((f r) y)
    ne : Not (Eq (Module.eqIdeal R x y) Top.top)
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑(Module.eqIdeal R x y) ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    n : Nat
    eq : Eq (HSMul.hSMul ⟨(fun x => HPow.hPow (↑r) x) n, ⋯⟩ x) (HSMul.hSMul ⟨(fun  …
    ⊢ False
  -/
  exact Set.disjoint_left.mp disj eq ⟨n, rfl⟩
  /-
    🎉 no goals
  -/


theorem Module.eq_zero_of_isLocalized_span (x : M) (h : ∀ r : s, f r x = 0) : x = 0 :=
                                                 /-
                                                   R : Type u_1
                                                   M : Type u_2
                                                   inst✝⁵ : CommSemiring R
                                                   inst✝⁴ : AddCommMonoid M
                                                   inst✝³ : Module R M
                                                   s : Set R
                                                   span_eq : Eq (Ideal.span s) Top.top
                                                   Mₚ : ↑s → Type u_5
                                                   inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                   inst✝¹ : (r : ↑s) → Module R (Mₚ r)
                                                   f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                   inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                   x : M
                                                   h : ∀ (r : ↑s), Eq ((f r) x) 0
                                                   ⊢ ∀ (r : ↑s), Eq ((f r) x) ((f r) 0)
                                                 -/
  eq_of_isLocalized_span s span_eq _ f x 0 <| by simpa only [map_zero] using h
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem Submodule.mem_of_isLocalized_span {m : M} {N : Submodule R M}
    (h : ∀ r : s, f r m ∈ N.localized₀ (.powers r.1) (f r)) : m ∈ N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    ⊢ Membership.mem N m
  -/
  let I : Ideal R := N.comap (LinearMap.toSpanSingleton R M m)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ⊢ Membership.mem N m
  -/
  suffices I = ⊤ by simpa [I] using I.eq_top_iff_one.mp this
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ⊢ Eq I Top.top
  -/
  by_contra! ne
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    ⊢ False
  -/
  have ⟨r, hrs, disj⟩ := exists_disjoint_powers_of_span_eq_top s span_eq _ ne
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r : R
    hrs : Membership.mem s r
    disj : Disjoint ↑I ↑(Submonoid.powers r)
    ⊢ False
  -/
  let r : s := ⟨r, hrs⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑I ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    ⊢ False
  -/
  obtain ⟨a, ha, t, e⟩ := h r
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑I ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    a : M
    ha : Membership.mem N a
    t : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    e : Eq (IsLocalizedModule.mk' (f r) a t) ((f r) m)
    ⊢ False
  -/
  rw [← IsLocalizedModule.mk'_one (.powers r.1), IsLocalizedModule.mk'_eq_mk'_iff] at e
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑I ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    a : M
    ha : Membership.mem N a
    t : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    e : Exists fun s_1 => Eq (HSMul.hSMul s_1 (HSMul.hSMul t m)) (HSMul.hSMul s_1  …
    ⊢ False
  -/
  have ⟨u, hu⟩ := e
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑I ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    a : M
    ha : Membership.mem N a
    t : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    e : Exists fun s_1 => Eq (HSMul.hSMul s_1 (HSMul.hSMul t m)) (HSMul.hSMul s_1  …
    u : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    hu : Eq (HSMul.hSMul u (HSMul.hSMul t m)) (HSMul.hSMul u (HSMul.hSMul 1 a))
    ⊢ False
  -/
  simp_rw [smul_smul] at hu
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    s : Set R
    span_eq : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝¹ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    m : M
    N : Submodule R M
    h : ∀ (r : ↑s), Membership.mem (Submodule.localized₀ (Submonoid.powers ↑r) (f  …
    I : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R M m) N
    ne : Ne I Top.top
    r✝ : R
    hrs : Membership.mem s r✝
    disj : Disjoint ↑I ↑(Submonoid.powers r✝)
    r : ↑s := ⟨r✝, hrs⟩
    a : M
    ha : Membership.mem N a
    t : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    e : Exists fun s_1 => Eq (HSMul.hSMul s_1 (HSMul.hSMul t m)) (HSMul.hSMul s_1  …
    u : Subtype fun x => Membership.mem (Submonoid.powers ↑r) x
    hu : Eq (HSMul.hSMul (HMul.hMul u t) m) (HSMul.hSMul (HMul.hMul u 1) a)
    ⊢ False
  -/
  exact Set.disjoint_right.mp disj (u * t).2 (by apply hu ▸ smul_mem _ _ ha)
  /-
    🎉 no goals
  -/


theorem Submodule.le_of_isLocalized_span {N P : Submodule R M}
    (h : ∀ r : s, N.localized₀ (.powers r.1) (f r) ≤ P.localized₀ (.powers r.1) (f r)) : N ≤ P :=
                                                                             /-
                                                                               R : Type u_1
                                                                               M : Type u_2
                                                                               inst✝⁵ : CommSemiring R
                                                                               inst✝⁴ : AddCommMonoid M
                                                                               inst✝³ : Module R M
                                                                               s : Set R
                                                                               span_eq : Eq (Ideal.span s) Top.top
                                                                               Mₚ : ↑s → Type u_5
                                                                               inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                                               inst✝¹ : (r : ↑s) → Module R (Mₚ r)
                                                                               f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                                               inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                                               N P : Submodule R M
                                                                               h : ∀ (r : ↑s), LE.le (Submodule.localized₀ (Submonoid.powers ↑r) (f r) N) (Su …
                                                                               m : M
                                                                               hm : Membership.mem N m
                                                                               r : ↑s
                                                                               ⊢ Eq (IsLocalizedModule.mk' (f r) m 1) ((f r) m)
                                                                             -/
  fun m hm ↦ mem_of_isLocalized_span s span_eq _ f fun r ↦ h r ⟨m, hm, 1, by simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem Submodule.eq_of_isLocalized₀_span {N P : Submodule R M}
    (h : ∀ r : s, N.localized₀ (.powers r.1) (f r) = P.localized₀ (.powers r.1) (f r)) : N = P :=
  le_antisymm (le_of_isLocalized_span s span_eq _ _ fun r ↦ (h r).le)
    (le_of_isLocalized_span s span_eq _ _ fun r ↦ (h r).ge)


theorem Submodule.eq_bot_of_isLocalized₀_span {N : Submodule R M}
    (h : ∀ r : s, N.localized₀ (.powers r.1) (f r) = ⊥) : N = ⊥ :=
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_2
                                                      inst✝⁵ : CommSemiring R
                                                      inst✝⁴ : AddCommMonoid M
                                                      inst✝³ : Module R M
                                                      s : Set R
                                                      span_eq : Eq (Ideal.span s) Top.top
                                                      Mₚ : ↑s → Type u_5
                                                      inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                      inst✝¹ : (r : ↑s) → Module R (Mₚ r)
                                                      f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                      inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                      N : Submodule R M
                                                      h : ∀ (r : ↑s), Eq (Submodule.localized₀ (Submonoid.powers ↑r) (f r) N) Bot.bot
                                                      x✝ : ↑s
                                                      ⊢ Eq (Submodule.localized₀ (Submonoid.powers ↑x✝) (f x✝) N) (Submodule.localiz …
                                                    -/
  eq_of_isLocalized₀_span s span_eq Mₚ f fun _ ↦ by simp only [h, Submodule.localized₀_bot]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem Submodule.eq_top_of_isLocalized₀_span {N : Submodule R M}
    (h : ∀ r : s, N.localized₀ (.powers r.1) (f r) = ⊤) : N = ⊤ :=
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_2
                                                      inst✝⁵ : CommSemiring R
                                                      inst✝⁴ : AddCommMonoid M
                                                      inst✝³ : Module R M
                                                      s : Set R
                                                      span_eq : Eq (Ideal.span s) Top.top
                                                      Mₚ : ↑s → Type u_5
                                                      inst✝² : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                      inst✝¹ : (r : ↑s) → Module R (Mₚ r)
                                                      f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                      inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                      N : Submodule R M
                                                      h : ∀ (r : ↑s), Eq (Submodule.localized₀ (Submonoid.powers ↑r) (f r) N) Top.top
                                                      x✝ : ↑s
                                                      ⊢ Eq (Submodule.localized₀ (Submonoid.powers ↑x✝) (f x✝) N) (Submodule.localiz …
                                                    -/
  eq_of_isLocalized₀_span s span_eq Mₚ f fun _ ↦ by simp only [h, Submodule.localized₀_top]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem Submodule.eq_of_isLocalized'_span {N P : Submodule R M}
    (h : ∀ r, N.localized' (Rₚ r) (.powers r.1) (f r) = P.localized' (Rₚ r) (.powers r.1) (f r)) :
    N = P :=
  eq_of_isLocalized₀_span s span_eq _ f fun r ↦ congr(restrictScalars _ $(h r))


theorem Submodule.eq_bot_of_isLocalized'_span {N : Submodule R M}
    (h : ∀ r : s, N.localized' (Rₚ r) (.powers r.1) (f r) = ⊥) : N = ⊥ :=
                                                       /-
                                                         R : Type u_1
                                                         M : Type u_2
                                                         inst✝¹⁰ : CommSemiring R
                                                         inst✝⁹ : AddCommMonoid M
                                                         inst✝⁸ : Module R M
                                                         s : Set R
                                                         span_eq : Eq (Ideal.span s) Top.top
                                                         Rₚ : ↑s → Type u_4
                                                         inst✝⁷ : (r : ↑s) → CommSemiring (Rₚ r)
                                                         inst✝⁶ : (r : ↑s) → Algebra R (Rₚ r)
                                                         inst✝⁵ : ∀ (r : ↑s), IsLocalization.Away (↑r) (Rₚ r)
                                                         Mₚ : ↑s → Type u_5
                                                         inst✝⁴ : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                         inst✝³ : (r : ↑s) → Module R (Mₚ r)
                                                         inst✝² : (r : ↑s) → Module (Rₚ r) (Mₚ r)
                                                         inst✝¹ : ∀ (r : ↑s), IsScalarTower R (Rₚ r) (Mₚ r)
                                                         f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                         inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                         N : Submodule R M
                                                         h : ∀ (r : ↑s), Eq (Submodule.localized' (Rₚ r) (Submonoid.powers ↑r) (f r) N) …
                                                         x✝ : ↑s
                                                         ⊢ Eq (Submodule.localized' (Rₚ x✝) (Submonoid.powers ↑x✝) (f x✝) N) (Submodule …
                                                       -/
  eq_of_isLocalized'_span s span_eq Rₚ Mₚ f fun _ ↦ by simp only [h, Submodule.localized'_bot]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Submodule.eq_top_of_isLocalized'_span {N : Submodule R M}
    (h : ∀ r : s, N.localized' (Rₚ r) (.powers r.1) (f r) = ⊤) : N = ⊤ :=
                                                       /-
                                                         R : Type u_1
                                                         M : Type u_2
                                                         inst✝¹⁰ : CommSemiring R
                                                         inst✝⁹ : AddCommMonoid M
                                                         inst✝⁸ : Module R M
                                                         s : Set R
                                                         span_eq : Eq (Ideal.span s) Top.top
                                                         Rₚ : ↑s → Type u_4
                                                         inst✝⁷ : (r : ↑s) → CommSemiring (Rₚ r)
                                                         inst✝⁶ : (r : ↑s) → Algebra R (Rₚ r)
                                                         inst✝⁵ : ∀ (r : ↑s), IsLocalization.Away (↑r) (Rₚ r)
                                                         Mₚ : ↑s → Type u_5
                                                         inst✝⁴ : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                         inst✝³ : (r : ↑s) → Module R (Mₚ r)
                                                         inst✝² : (r : ↑s) → Module (Rₚ r) (Mₚ r)
                                                         inst✝¹ : ∀ (r : ↑s), IsScalarTower R (Rₚ r) (Mₚ r)
                                                         f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                         inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                         N : Submodule R M
                                                         h : ∀ (r : ↑s), Eq (Submodule.localized' (Rₚ r) (Submonoid.powers ↑r) (f r) N) …
                                                         x✝ : ↑s
                                                         ⊢ Eq (Submodule.localized' (Rₚ x✝) (Submonoid.powers ↑x✝) (f x✝) N) (Submodule …
                                                       -/
  eq_of_isLocalized'_span s span_eq Rₚ Mₚ f fun _ ↦ by simp only [h, Submodule.localized'_top]
                                                       /-
                                                         🎉 no goals
                                                       -/



/-- The normalizer of a Lie submodule.

See also `LieSubmodule.idealizer`. -/
def normalizer : LieSubmodule R L M where
  carrier := {m | ∀ x : L, ⁅x, m⁆ ∈ N}
                           /-
                             R : Type u_1
                             L : Type u_2
                             M : Type u_3
                             M' : Type u_4
                             inst✝¹⁰ : CommRing R
                             inst✝⁹ : LieRing L
                             inst✝⁸ : LieAlgebra R L
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : LieRingModule L M
                             inst✝⁴ : LieModule R L M
                             inst✝³ : AddCommGroup M'
                             inst✝² : Module R M'
                             inst✝¹ : LieRingModule L M'
                             inst✝ : LieModule R L M'
                             N N₁ N₂ : LieSubmodule R L M
                             a✝ b✝ : M
                             hm₁ : Membership.mem (setOf fun m => ∀ (x : L), Membership.mem N (Bracket.brac …
                             hm₂ : Membership.mem (setOf fun m => ∀ (x : L), Membership.mem N (Bracket.brac …
                             x : L
                             ⊢ Membership.mem N (Bracket.bracket x (HAdd.hAdd a✝ b✝))
                           -/
  add_mem' hm₁ hm₂ x := by rw [lie_add]; exact N.add_mem' (hm₁ x) (hm₂ x)
                                         /-
                                           🎉 no goals
                                         -/
                    /-
                      R : Type u_1
                      L : Type u_2
                      M : Type u_3
                      M' : Type u_4
                      inst✝¹⁰ : CommRing R
                      inst✝⁹ : LieRing L
                      inst✝⁸ : LieAlgebra R L
                      inst✝⁷ : AddCommGroup M
                      inst✝⁶ : Module R M
                      inst✝⁵ : LieRingModule L M
                      inst✝⁴ : LieModule R L M
                      inst✝³ : AddCommGroup M'
                      inst✝² : Module R M'
                      inst✝¹ : LieRingModule L M'
                      inst✝ : LieModule R L M'
                      N N₁ N₂ : LieSubmodule R L M
                      x : L
                      ⊢ Membership.mem N (Bracket.bracket x 0)
                    -/
  zero_mem' x := by simp
                    /-
                      🎉 no goals
                    -/
                           /-
                             R : Type u_1
                             L : Type u_2
                             M : Type u_3
                             M' : Type u_4
                             inst✝¹⁰ : CommRing R
                             inst✝⁹ : LieRing L
                             inst✝⁸ : LieAlgebra R L
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : LieRingModule L M
                             inst✝⁴ : LieModule R L M
                             inst✝³ : AddCommGroup M'
                             inst✝² : Module R M'
                             inst✝¹ : LieRingModule L M'
                             inst✝ : LieModule R L M'
                             N N₁ N₂ : LieSubmodule R L M
                             t : R
                             m : M
                             hm : Membership.mem { carrier := setOf fun m => ∀ (x : L), Membership.mem N (B …
                             x : L
                             ⊢ Membership.mem N (Bracket.bracket x (HSMul.hSMul t m))
                           -/
  smul_mem' t m hm x := by rw [lie_smul]; exact N.smul_mem' t (hm x)
                                          /-
                                            🎉 no goals
                                          -/
                           /-
                             R : Type u_1
                             L : Type u_2
                             M : Type u_3
                             M' : Type u_4
                             inst✝¹⁰ : CommRing R
                             inst✝⁹ : LieRing L
                             inst✝⁸ : LieAlgebra R L
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : LieRingModule L M
                             inst✝⁴ : LieModule R L M
                             inst✝³ : AddCommGroup M'
                             inst✝² : Module R M'
                             inst✝¹ : LieRingModule L M'
                             inst✝ : LieModule R L M'
                             N N₁ N₂ : LieSubmodule R L M
                             x : L
                             m : M
                             hm : Membership.mem { carrier := setOf fun m => ∀ (x : L), Membership.mem N (B …
                             y : L
                             ⊢ Membership.mem N (Bracket.bracket y (Bracket.bracket x m))
                           -/
  lie_mem {x m} hm y := by rw [leibniz_lie]; exact N.add_mem' (hm ⁅y, x⁆) (N.lie_mem (hm y))
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem mem_normalizer (m : M) : m ∈ N.normalizer ↔ ∀ x : L, ⁅x, m⁆ ∈ N :=
  Iff.rfl


@[simp]
theorem le_normalizer : N ≤ N.normalizer := by
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
    N : LieSubmodule R L M
    ⊢ LE.le N N.normalizer
  -/
  intro m hm
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
    N : LieSubmodule R L M
    m : M
    hm : Membership.mem N m
    ⊢ Membership.mem N.normalizer m
  -/
  rw [mem_normalizer]
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
    N : LieSubmodule R L M
    m : M
    hm : Membership.mem N m
    ⊢ ∀ (x : L), Membership.mem N (Bracket.bracket x m)
  -/
  exact fun x => N.lie_mem hm
  /-
    🎉 no goals
  -/


theorem normalizer_inf : (N₁ ⊓ N₂).normalizer = N₁.normalizer ⊓ N₂.normalizer := by
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
    N₁ N₂ : LieSubmodule R L M
    ⊢ Eq (Min.min N₁ N₂).normalizer (Min.min N₁.normalizer N₂.normalizer)
  -/
  ext; simp [← forall_and]
       /-
         🎉 no goals
       -/


@[gcongr, mono]
theorem normalizer_mono (h : N₁ ≤ N₂) : normalizer N₁ ≤ normalizer N₂ := by
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
    N₁ N₂ : LieSubmodule R L M
    h : LE.le N₁ N₂
    ⊢ LE.le N₁.normalizer N₂.normalizer
  -/
  intro m hm
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
    N₁ N₂ : LieSubmodule R L M
    h : LE.le N₁ N₂
    m : M
    hm : Membership.mem N₁.normalizer m
    ⊢ Membership.mem N₂.normalizer m
  -/
  rw [mem_normalizer] at hm ⊢
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
    N₁ N₂ : LieSubmodule R L M
    h : LE.le N₁ N₂
    m : M
    hm : ∀ (x : L), Membership.mem N₁ (Bracket.bracket x m)
    ⊢ ∀ (x : L), Membership.mem N₂ (Bracket.bracket x m)
  -/
  exact fun x ↦ h (hm x)
  /-
    🎉 no goals
  -/


theorem monotone_normalizer : Monotone (normalizer : LieSubmodule R L M → LieSubmodule R L M) :=
  fun _ _ ↦ normalizer_mono


@[simp]
theorem comap_normalizer (f : M' →ₗ⁅R,L⁆ M) : N.normalizer.comap f = (N.comap f).normalizer := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    M' : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    inst✝¹ : LieRingModule L M'
    inst✝ : LieModule R L M'
    N : LieSubmodule R L M
    f : LieModuleHom R L M' M
    ⊢ Eq (LieSubmodule.comap f N.normalizer) (LieSubmodule.comap f N).normalizer
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem top_lie_le_iff_le_normalizer (N' : LieSubmodule R L M) :
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
                                                             N N' : LieSubmodule R L M
                                                             ⊢ Iff (LE.le (Bracket.bracket Top.top N) N') (LE.le N N'.normalizer)
                                                           -/
    ⁅(⊤ : LieIdeal R L), N⁆ ≤ N' ↔ N ≤ N'.normalizer := by rw [lie_le_iff]; tauto
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem gc_top_lie_normalizer :
    GaloisConnection (fun N : LieSubmodule R L M => ⁅(⊤ : LieIdeal R L), N⁆) normalizer :=
  top_lie_le_iff_le_normalizer


variable (R L M) in
theorem normalizer_bot_eq_maxTrivSubmodule :
    (⊥ : LieSubmodule R L M).normalizer = LieModule.maxTrivSubmodule R L M :=
  rfl


/-- The idealizer of a Lie submodule.

See also `LieSubmodule.normalizer`. -/
def idealizer : LieIdeal R L where
  carrier := {x : L | ∀ m : M, ⁅x, m⁆ ∈ N}
                                       /-
                                         R : Type u_1
                                         L : Type u_2
                                         M : Type u_3
                                         M' : Type u_4
                                         inst✝¹⁰ : CommRing R
                                         inst✝⁹ : LieRing L
                                         inst✝⁸ : LieAlgebra R L
                                         inst✝⁷ : AddCommGroup M
                                         inst✝⁶ : Module R M
                                         inst✝⁵ : LieRingModule L M
                                         inst✝⁴ : LieModule R L M
                                         inst✝³ : AddCommGroup M'
                                         inst✝² : Module R M'
                                         inst✝¹ : LieRingModule L M'
                                         inst✝ : LieModule R L M'
                                         N N₁ N₂ : LieSubmodule R L M
                                         x y : L
                                         hx : Membership.mem (setOf fun x => ∀ (m : M), Membership.mem N (Bracket.brack …
                                         hy : Membership.mem (setOf fun x => ∀ (m : M), Membership.mem N (Bracket.brack …
                                         m : M
                                         ⊢ Membership.mem N (Bracket.bracket (HAdd.hAdd x y) m)
                                       -/
  add_mem' := fun {x} {y} hx hy m ↦ by rw [add_lie]; exact N.add_mem (hx m) (hy m)
                                                     /-
                                                       🎉 no goals
                                                     -/
                  /-
                    R : Type u_1
                    L : Type u_2
                    M : Type u_3
                    M' : Type u_4
                    inst✝¹⁰ : CommRing R
                    inst✝⁹ : LieRing L
                    inst✝⁸ : LieAlgebra R L
                    inst✝⁷ : AddCommGroup M
                    inst✝⁶ : Module R M
                    inst✝⁵ : LieRingModule L M
                    inst✝⁴ : LieModule R L M
                    inst✝³ : AddCommGroup M'
                    inst✝² : Module R M'
                    inst✝¹ : LieRingModule L M'
                    inst✝ : LieModule R L M'
                    N N₁ N₂ : LieSubmodule R L M
                    ⊢ Membership.mem { carrier := setOf fun x => ∀ (m : M), Membership.mem N (Brac …
                  -/
  zero_mem' := by simp
                  /-
                    🎉 no goals
                  -/
                                   /-
                                     R : Type u_1
                                     L : Type u_2
                                     M : Type u_3
                                     M' : Type u_4
                                     inst✝¹⁰ : CommRing R
                                     inst✝⁹ : LieRing L
                                     inst✝⁸ : LieAlgebra R L
                                     inst✝⁷ : AddCommGroup M
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : LieRingModule L M
                                     inst✝⁴ : LieModule R L M
                                     inst✝³ : AddCommGroup M'
                                     inst✝² : Module R M'
                                     inst✝¹ : LieRingModule L M'
                                     inst✝ : LieModule R L M'
                                     N N₁ N₂ : LieSubmodule R L M
                                     t : R
                                     x : L
                                     hx : Membership.mem { carrier := setOf fun x => ∀ (m : M), Membership.mem N (B …
                                     m : M
                                     ⊢ Membership.mem N (Bracket.bracket (HSMul.hSMul t x) m)
                                   -/
  smul_mem' := fun t {x} hx m ↦ by rw [smul_lie]; exact N.smul_mem t (hx m)
                                                  /-
                                                    🎉 no goals
                                                  -/
                                   /-
                                     R : Type u_1
                                     L : Type u_2
                                     M : Type u_3
                                     M' : Type u_4
                                     inst✝¹⁰ : CommRing R
                                     inst✝⁹ : LieRing L
                                     inst✝⁸ : LieAlgebra R L
                                     inst✝⁷ : AddCommGroup M
                                     inst✝⁶ : Module R M
                                     inst✝⁵ : LieRingModule L M
                                     inst✝⁴ : LieModule R L M
                                     inst✝³ : AddCommGroup M'
                                     inst✝² : Module R M'
                                     inst✝¹ : LieRingModule L M'
                                     inst✝ : LieModule R L M'
                                     N N₁ N₂ : LieSubmodule R L M
                                     x y : L
                                     hy : Membership.mem { carrier := setOf fun x => ∀ (m : M), Membership.mem N (B …
                                     m : M
                                     ⊢ Membership.mem N (Bracket.bracket (Bracket.bracket x y) m)
                                   -/
  lie_mem := fun {x} {y} hy m ↦ by rw [lie_lie]; exact sub_mem (N.lie_mem (hy m)) (hy ⁅x, m⁆)
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
lemma mem_idealizer {x : L} : x ∈ N.idealizer ↔ ∀ m : M, ⁅x, m⁆ ∈ N := Iff.rfl


@[simp]
lemma _root_.LieIdeal.idealizer_eq_normalizer (I : LieIdeal R L) :
    I.idealizer = I.normalizer := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Eq (LieSubmodule.idealizer I) (LieSubmodule.normalizer I)
  -/
  ext x; exact forall_congr' fun y ↦ by simp only [← lie_skew x y, neg_mem_iff]
         /-
           🎉 no goals
         -/


/-- Regarding a Lie subalgebra `H ⊆ L` as a module over itself, its normalizer is in fact a Lie
subalgebra. -/
def normalizer : LieSubalgebra R L :=
  { H.toLieSubmodule.normalizer with
    lie_mem' := fun {y z} hy hz x => by
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        M' : Type u_4
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : LieAlgebra R L
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : LieRingModule L M
        inst✝⁴ : LieModule R L M
        inst✝³ : AddCommGroup M'
        inst✝² : Module R M'
        inst✝¹ : LieRingModule L M'
        inst✝ : LieModule R L M'
        H : LieSubalgebra R L
        y z : L
        hy : Membership.mem (↑__src✝).carrier y
        hz : Membership.mem (↑__src✝).carrier z
        x : Subtype fun x => Membership.mem H x
        ⊢ Membership.mem H.toLieSubmodule (Bracket.bracket x (Bracket.bracket y z))
      -/
      rw [coe_bracket_of_module, mem_toLieSubmodule, leibniz_lie, ← lie_skew y, ← sub_eq_add_neg]
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        M' : Type u_4
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : LieAlgebra R L
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : LieRingModule L M
        inst✝⁴ : LieModule R L M
        inst✝³ : AddCommGroup M'
        inst✝² : Module R M'
        inst✝¹ : LieRingModule L M'
        inst✝ : LieModule R L M'
        H : LieSubalgebra R L
        y z : L
        hy : Membership.mem (↑__src✝).carrier y
        hz : Membership.mem (↑__src✝).carrier z
        x : Subtype fun x => Membership.mem H x
        ⊢ Membership.mem H (HSub.hSub (Bracket.bracket (Bracket.bracket (↑x) y) z) (Br …
      -/
      exact H.sub_mem (hz ⟨_, hy x⟩) (hy ⟨_, hz x⟩) }
      /-
        🎉 no goals
      -/


theorem mem_normalizer_iff' (x : L) : x ∈ H.normalizer ↔ ∀ y : L, y ∈ H → ⁅y, x⁆ ∈ H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    ⊢ Iff (Membership.mem H.normalizer x) (∀ (y : L), Membership.mem H y → Members …
  -/
  rw [Subtype.forall']; rfl
                        /-
                          🎉 no goals
                        -/


theorem mem_normalizer_iff (x : L) : x ∈ H.normalizer ↔ ∀ y : L, y ∈ H → ⁅x, y⁆ ∈ H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    ⊢ Iff (Membership.mem H.normalizer x) (∀ (y : L), Membership.mem H y → Members …
  -/
  rw [mem_normalizer_iff']
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    ⊢ Iff (∀ (y : L), Membership.mem H y → Membership.mem H (Bracket.bracket y x)) …
  -/
  refine forall₂_congr fun y hy => ?_
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x y : L
    hy : Membership.mem H y
    ⊢ Iff (Membership.mem H (Bracket.bracket y x)) (Membership.mem H (Bracket.brac …
  -/
  rw [← lie_skew, neg_mem_iff (G := L)]
  /-
    🎉 no goals
  -/


theorem le_normalizer : H ≤ H.normalizer :=
  H.toLieSubmodule.le_normalizer


theorem coe_normalizer_eq_normalizer :
    (H.toLieSubmodule.normalizer : Submodule R L) = H.normalizer :=
  rfl


theorem lie_mem_sup_of_mem_normalizer {x y z : L} (hx : x ∈ H.normalizer) (hy : y ∈ (R ∙ x) ⊔ ↑H)
    (hz : z ∈ (R ∙ x) ⊔ ↑H) : ⁅y, z⁆ ∈ (R ∙ x) ⊔ ↑H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x y z : L
    hx : Membership.mem H.normalizer x
    hy : Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSu …
    hz : Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSu …
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  rw [Submodule.mem_sup] at hy hz
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x y z : L
    hx : Membership.mem H.normalizer x
    hy : Exists fun y_1 => And (Membership.mem (Submodule.span R (Singleton.single …
    hz : Exists fun y => And (Membership.mem (Submodule.span R (Singleton.singleto …
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  obtain ⟨u₁, hu₁, v, hv : v ∈ H, rfl⟩ := hy
  /-
    case intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x z : L
    hx : Membership.mem H.normalizer x
    hz : Exists fun y => And (Membership.mem (Submodule.span R (Singleton.singleto …
    u₁ : L
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) u₁
    v : L
    hv : Membership.mem H v
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  obtain ⟨u₂, hu₂, w, hw : w ∈ H, rfl⟩ := hz
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    hx : Membership.mem H.normalizer x
    u₁ : L
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) u₁
    v : L
    hv : Membership.mem H v
    u₂ : L
    hu₂ : Membership.mem (Submodule.span R (Singleton.singleton x)) u₂
    w : L
    hw : Membership.mem H w
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  obtain ⟨t, rfl⟩ := Submodule.mem_span_singleton.mp hu₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    hx : Membership.mem H.normalizer x
    v : L
    hv : Membership.mem H v
    u₂ : L
    hu₂ : Membership.mem (Submodule.span R (Singleton.singleton x)) u₂
    w : L
    hw : Membership.mem H w
    t : R
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul t …
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  obtain ⟨s, rfl⟩ := Submodule.mem_span_singleton.mp hu₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    hx : Membership.mem H.normalizer x
    v : L
    hv : Membership.mem H v
    w : L
    hw : Membership.mem H w
    t : R
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul t …
    s : R
    hu₂ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul s …
    ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) H.toSubmo …
  -/
  apply Submodule.mem_sup_right
  simp only [LieSubalgebra.mem_toSubmodule, smul_lie, add_lie, zero_add, lie_add, smul_zero,
    lie_smul, lie_self]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.a
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    hx : Membership.mem H.normalizer x
    v : L
    hv : Membership.mem H v
    w : L
    hw : Membership.mem H w
    t : R
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul t …
    s : R
    hu₂ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul s …
    ⊢ Membership.mem H (HAdd.hAdd (HSMul.hSMul s (Bracket.bracket v x)) (HAdd.hAdd …
  -/
  refine H.add_mem (H.smul_mem s ?_) (H.add_mem (H.smul_mem t ?_) (H.lie_mem hv hw))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.refine_1
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x : L
    hx : Membership.mem H.normalizer x
    v : L
    hv : Membership.mem H v
    w : L
    hw : Membership.mem H w
    t : R
    hu₁ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul t …
    s : R
    hu₂ : Membership.mem (Submodule.span R (Singleton.singleton x)) (HSMul.hSMul s …
    ⊢ Membership.mem H (Bracket.bracket v x)
  -/
  exacts [(H.mem_normalizer_iff' x).mp hx v hv, (H.mem_normalizer_iff x).mp hx w hw]
  /-
    🎉 no goals
  -/


/-- A Lie subalgebra is an ideal of its normalizer. -/
theorem ideal_in_normalizer {x y : L} (hx : x ∈ H.normalizer) (hy : y ∈ H) : ⁅x, y⁆ ∈ H := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x y : L
    hx : Membership.mem H.normalizer x
    hy : Membership.mem H y
    ⊢ Membership.mem H (Bracket.bracket x y)
  -/
  rw [← lie_skew, neg_mem_iff (G := L)]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    x y : L
    hx : Membership.mem H.normalizer x
    hy : Membership.mem H y
    ⊢ Membership.mem H (Bracket.bracket y x)
  -/
  exact hx ⟨y, hy⟩
  /-
    🎉 no goals
  -/


/-- A Lie subalgebra `H` is an ideal of any Lie subalgebra `K` containing `H` and contained in the
normalizer of `H`. -/
theorem exists_nested_lieIdeal_ofLe_normalizer {K : LieSubalgebra R L} (h₁ : H ≤ K)
    (h₂ : K ≤ H.normalizer) : ∃ I : LieIdeal R K, (I : LieSubalgebra R K) = ofLe h₁ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H K : LieSubalgebra R L
    h₁ : LE.le H K
    h₂ : LE.le K H.normalizer
    ⊢ Exists fun I => Eq (LieIdeal.toLieSubalgebra R (Subtype fun x => Membership. …
  -/
  rw [exists_nested_lieIdeal_coe_eq_iff]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H K : LieSubalgebra R L
    h₁ : LE.le H K
    h₂ : LE.le K H.normalizer
    ⊢ ∀ (x y : L), Membership.mem K x → Membership.mem H y → Membership.mem H (Bra …
  -/
  exact fun x y hx hy => ideal_in_normalizer (h₂ hx) hy
  /-
    🎉 no goals
  -/


theorem normalizer_eq_self_iff :
    H.normalizer = H ↔ (LieModule.maxTrivSubmodule R H <| L ⧸ H.toLieSubmodule) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    ⊢ Iff (Eq H.normalizer H) (Eq (LieModule.maxTrivSubmodule R (Subtype fun x =>  …
  -/
  rw [LieSubmodule.eq_bot_iff]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    H : LieSubalgebra R L
    ⊢ Iff (Eq H.normalizer H) (∀ (m : HasQuotient.Quotient L H.toLieSubmodule), Me …
  -/
  refine ⟨fun h => ?_, fun h => le_antisymm ?_ H.le_normalizer⟩
    /-
      case refine_1
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : Eq H.normalizer H
      ⊢ ∀ (m : HasQuotient.Quotient L H.toLieSubmodule), Membership.mem (LieModule.m …
    -/
  · rintro ⟨x⟩ hx
    suffices x ∈ H by rwa [Submodule.Quotient.quot_mk_eq_mk, Submodule.Quotient.mk_eq_zero,
      coe_toLieSubmodule, mem_toSubmodule]
    /-
      case refine_1.mk
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : Eq H.normalizer H
      m✝ : HasQuotient.Quotient L H.toLieSubmodule
      x : L
      hx : Membership.mem (LieModule.maxTrivSubmodule R (Subtype fun x => Membership …
      ⊢ Membership.mem H x
    -/
    rw [← h, H.mem_normalizer_iff']
    /-
      case refine_1.mk
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : Eq H.normalizer H
      m✝ : HasQuotient.Quotient L H.toLieSubmodule
      x : L
      hx : Membership.mem (LieModule.maxTrivSubmodule R (Subtype fun x => Membership …
      ⊢ ∀ (y : L), Membership.mem H y → Membership.mem H (Bracket.bracket y x)
    -/
    intro y hy
    /-
      case refine_1.mk
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : Eq H.normalizer H
      m✝ : HasQuotient.Quotient L H.toLieSubmodule
      x : L
      hx : Membership.mem (LieModule.maxTrivSubmodule R (Subtype fun x => Membership …
      y : L
      hy : Membership.mem H y
      ⊢ Membership.mem H (Bracket.bracket y x)
    -/
    replace hx : ⁅_, LieSubmodule.Quotient.mk' _ x⁆ = 0 := hx ⟨y, hy⟩
    /-
      case refine_1.mk
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : Eq H.normalizer H
      m✝ : HasQuotient.Quotient L H.toLieSubmodule
      x y : L
      hy : Membership.mem H y
      hx : Eq (Bracket.bracket ⟨y, hy⟩ ((LieSubmodule.Quotient.mk' H.toLieSubmodule) …
      ⊢ Membership.mem H (Bracket.bracket y x)
    -/
    rwa [← LieModuleHom.map_lie, LieSubmodule.Quotient.mk_eq_zero] at hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : ∀ (m : HasQuotient.Quotient L H.toLieSubmodule), Membership.mem (LieModule …
      ⊢ LE.le H.normalizer H
    -/
  · intro x hx
    /-
      case refine_2
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : ∀ (m : HasQuotient.Quotient L H.toLieSubmodule), Membership.mem (LieModule …
      x : L
      hx : Membership.mem H.normalizer x
      ⊢ Membership.mem H x
    -/
    let y := LieSubmodule.Quotient.mk' H.toLieSubmodule x
    have hy : y ∈ LieModule.maxTrivSubmodule R H (L ⧸ H.toLieSubmodule) := by
      rintro ⟨z, hz⟩
      rw [← LieModuleHom.map_lie, LieSubmodule.Quotient.mk_eq_zero, coe_bracket_of_module,
        Submodule.coe_mk, mem_toLieSubmodule]
      exact (H.mem_normalizer_iff' x).mp hx z hz
    /-
      case refine_2
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      H : LieSubalgebra R L
      h : ∀ (m : HasQuotient.Quotient L H.toLieSubmodule), Membership.mem (LieModule …
      x : L
      hx : Membership.mem H.normalizer x
      y : HasQuotient.Quotient L H.toLieSubmodule := (LieSubmodule.Quotient.mk' H.to …
      hy : Membership.mem (LieModule.maxTrivSubmodule R (Subtype fun x => Membership …
      ⊢ Membership.mem H x
    -/
    simpa [y] using h y hy
    /-
      🎉 no goals
    -/



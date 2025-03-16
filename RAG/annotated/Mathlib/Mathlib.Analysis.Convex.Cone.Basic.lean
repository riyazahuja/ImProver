/-- A convex cone is a subset `s` of a `𝕜`-module such that `a • x + b • y ∈ s` whenever `a, b > 0`
and `x, y ∈ s`. -/
structure ConvexCone [AddCommMonoid E] [SMul 𝕜 E] where
  /-- The **carrier set** underlying this cone: the set of points contained in it -/
  carrier : Set E
  smul_mem' : ∀ ⦃c : 𝕜⦄, 0 < c → ∀ ⦃x : E⦄, x ∈ carrier → c • x ∈ carrier
  add_mem' : ∀ ⦃x⦄ (_ : x ∈ carrier) ⦃y⦄ (_ : y ∈ carrier), x + y ∈ carrier


instance : SetLike (ConvexCone 𝕜 E) E where
  coe := carrier
                             /-
                               𝕜 : Type u_1
                               E : Type u_2
                               F : Type u_3
                               G : Type u_4
                               inst✝² : OrderedSemiring 𝕜
                               inst✝¹ : AddCommMonoid E
                               inst✝ : SMul 𝕜 E
                               S✝ T✝ S T : ConvexCone 𝕜 E
                               h : Eq S.carrier T.carrier
                               ⊢ Eq S T
                             -/
  coe_injective' S T h := by cases S; cases T; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem coe_mk {s : Set E} {h₁ h₂} : ↑(@mk 𝕜 _ _ _ _ s h₁ h₂) = s :=
  rfl


@[simp]
theorem mem_mk {s : Set E} {h₁ h₂ x} : x ∈ @mk 𝕜 _ _ _ _ s h₁ h₂ ↔ x ∈ s :=
  Iff.rfl


/-- Two `ConvexCone`s are equal if they have the same elements. -/
@[ext]
theorem ext {S T : ConvexCone 𝕜 E} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[aesop safe apply (rule_sets := [SetLike])]
theorem smul_mem {c : 𝕜} {x : E} (hc : 0 < c) (hx : x ∈ S) : c • x ∈ S :=
  S.smul_mem' hc hx


theorem add_mem ⦃x⦄ (hx : x ∈ S) ⦃y⦄ (hy : y ∈ S) : x + y ∈ S :=
  S.add_mem' hx hy


instance : AddMemClass (ConvexCone 𝕜 E) E where add_mem ha hb := add_mem _ ha hb


instance : Min (ConvexCone 𝕜 E) :=
  ⟨fun S T =>
    ⟨S ∩ T, fun _ hc _ hx => ⟨S.smul_mem hc hx.1, T.smul_mem hc hx.2⟩, fun _ hx _ hy =>
      ⟨S.add_mem hx.1 hy.1, T.add_mem hx.2 hy.2⟩⟩⟩


@[simp]
theorem coe_inf : ((S ⊓ T : ConvexCone 𝕜 E) : Set E) = ↑S ∩ ↑T :=
  rfl


theorem mem_inf {x} : x ∈ S ⊓ T ↔ x ∈ S ∧ x ∈ T :=
  Iff.rfl


instance : InfSet (ConvexCone 𝕜 E) :=
  ⟨fun S =>
    ⟨⋂ s ∈ S, ↑s, fun _ hc _ hx => mem_biInter fun s hs => s.smul_mem hc <| mem_iInter₂.1 hx s hs,
      fun _ hx _ hy =>
      mem_biInter fun s hs => s.add_mem (mem_iInter₂.1 hx s hs) (mem_iInter₂.1 hy s hs)⟩⟩


@[simp]
theorem coe_sInf (S : Set (ConvexCone 𝕜 E)) : ↑(sInf S) = ⋂ s ∈ S, (s : Set E) :=
  rfl


theorem mem_sInf {x : E} {S : Set (ConvexCone 𝕜 E)} : x ∈ sInf S ↔ ∀ s ∈ S, x ∈ s :=
  mem_iInter₂


@[simp]
theorem coe_iInf {ι : Sort*} (f : ι → ConvexCone 𝕜 E) : ↑(iInf f) = ⋂ i, (f i : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    ι : Sort u_5
    f : ι → ConvexCone 𝕜 E
    ⊢ Eq (↑(iInf f)) (Set.iInter fun i => ↑(f i))
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


theorem mem_iInf {ι : Sort*} {x : E} {f : ι → ConvexCone 𝕜 E} : x ∈ iInf f ↔ ∀ i, x ∈ f i :=
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            inst✝² : OrderedSemiring 𝕜
                            inst✝¹ : AddCommMonoid E
                            inst✝ : SMul 𝕜 E
                            ι : Sort u_5
                            x : E
                            f : ι → ConvexCone 𝕜 E
                            ⊢ Iff (∀ (i : ConvexCone 𝕜 E), Membership.mem (Set.range f) i → Membership.mem …
                          -/
  mem_iInter₂.trans <| by simp
                          /-
                            🎉 no goals
                          -/


instance : Bot (ConvexCone 𝕜 E) :=
  ⟨⟨∅, fun _ _ _ => False.elim, fun _ => False.elim⟩⟩


theorem mem_bot (x : E) : (x ∈ (⊥ : ConvexCone 𝕜 E)) = False :=
  rfl


@[simp]
theorem coe_bot : ↑(⊥ : ConvexCone 𝕜 E) = (∅ : Set E) :=
  rfl


instance : Top (ConvexCone 𝕜 E) :=
  ⟨⟨univ, fun _ _ _ _ => mem_univ _, fun _ _ _ _ => mem_univ _⟩⟩


theorem mem_top (x : E) : x ∈ (⊤ : ConvexCone 𝕜 E) :=
  mem_univ x


@[simp]
theorem coe_top : ↑(⊤ : ConvexCone 𝕜 E) = (univ : Set E) :=
  rfl


instance : CompleteLattice (ConvexCone 𝕜 E) :=
  { SetLike.instPartialOrder with
    le := (· ≤ ·)
    lt := (· < ·)
    bot := ⊥
    bot_le := fun _ _ => False.elim
    top := ⊤
    le_top := fun _ x _ => mem_top 𝕜 x
    inf := (· ⊓ ·)
    sInf := InfSet.sInf
    sup := fun a b => sInf { x | a ≤ x ∧ b ≤ x }
    sSup := fun s => sInf { T | ∀ S ∈ s, S ≤ T }
    le_sup_left := fun _ _ => fun _ hx => mem_sInf.2 fun _ hs => hs.1 hx
    le_sup_right := fun _ _ => fun _ hx => mem_sInf.2 fun _ hs => hs.2 hx
    sup_le := fun _ _ c ha hb _ hx => mem_sInf.1 hx c ⟨ha, hb⟩
    le_inf := fun _ _ _ ha hb _ hx => ⟨ha hx, hb hx⟩
    inf_le_left := fun _ _ _ => And.left
    inf_le_right := fun _ _ _ => And.right
    le_sSup := fun _ p hs _ hx => mem_sInf.2 fun _ ht => ht p hs hx
    sSup_le := fun _ p hs _ hx => mem_sInf.1 hx p hs
    le_sInf := fun _ _ ha _ hx => mem_sInf.2 fun t ht => ha t ht hx
    sInf_le := fun _ _ ha _ hx => mem_sInf.1 hx _ ha }


instance : Inhabited (ConvexCone 𝕜 E) :=
  ⟨⊥⟩


protected theorem convex : Convex 𝕜 (S : Set E) :=
  convex_iff_forall_pos.2 fun _ hx _ hy _ _ ha hb _ =>
    S.add_mem (S.smul_mem ha hx) (S.smul_mem hb hy)


/-- The image of a convex cone under a `𝕜`-linear map is a convex cone. -/
def map (f : E →ₗ[𝕜] F) (S : ConvexCone 𝕜 E) : ConvexCone 𝕜 F where
  carrier := f '' S
  smul_mem' := fun c hc _ ⟨x, hx, hy⟩ => hy ▸ f.map_smul c x ▸ mem_image_of_mem f (S.smul_mem hc hx)
  add_mem' := fun _ ⟨x₁, hx₁, hy₁⟩ _ ⟨x₂, hx₂, hy₂⟩ =>
    hy₁ ▸ hy₂ ▸ f.map_add x₁ x₂ ▸ mem_image_of_mem f (S.add_mem hx₁ hx₂)


@[simp, norm_cast]
theorem coe_map (S : ConvexCone 𝕜 E) (f : E →ₗ[𝕜] F) : (S.map f : Set F) = f '' S :=
  rfl


@[simp]
theorem mem_map {f : E →ₗ[𝕜] F} {S : ConvexCone 𝕜 E} {y : F} : y ∈ S.map f ↔ ∃ x ∈ S, f x = y :=
  Set.mem_image f S y


theorem map_map (g : F →ₗ[𝕜] G) (f : E →ₗ[𝕜] F) (S : ConvexCone 𝕜 E) :
    (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| image_image g f S


@[simp]
theorem map_id (S : ConvexCone 𝕜 E) : S.map LinearMap.id = S :=
  SetLike.coe_injective <| image_id _


/-- The preimage of a convex cone under a `𝕜`-linear map is a convex cone. -/
def comap (f : E →ₗ[𝕜] F) (S : ConvexCone 𝕜 F) : ConvexCone 𝕜 E where
  carrier := f ⁻¹' S
  smul_mem' c hc x hx := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : OrderedSemiring 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : AddCommMonoid G
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : Module 𝕜 G
      f : LinearMap (RingHom.id 𝕜) E F
      S : ConvexCone 𝕜 F
      c : 𝕜
      hc : LT.lt 0 c
      x : E
      hx : Membership.mem (Set.preimage ⇑f ↑S) x
      ⊢ Membership.mem (Set.preimage ⇑f ↑S) (HSMul.hSMul c x)
    -/
    rw [mem_preimage, f.map_smul c]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : OrderedSemiring 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : AddCommMonoid G
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : Module 𝕜 G
      f : LinearMap (RingHom.id 𝕜) E F
      S : ConvexCone 𝕜 F
      c : 𝕜
      hc : LT.lt 0 c
      x : E
      hx : Membership.mem (Set.preimage ⇑f ↑S) x
      ⊢ Membership.mem (↑S) (HSMul.hSMul c (f x))
    -/
    exact S.smul_mem hc hx
    /-
      🎉 no goals
    -/
  add_mem' x hx y hy := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : OrderedSemiring 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : AddCommMonoid G
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : Module 𝕜 G
      f : LinearMap (RingHom.id 𝕜) E F
      S : ConvexCone 𝕜 F
      x : E
      hx : Membership.mem (Set.preimage ⇑f ↑S) x
      y : E
      hy : Membership.mem (Set.preimage ⇑f ↑S) y
      ⊢ Membership.mem (Set.preimage ⇑f ↑S) (HAdd.hAdd x y)
    -/
    rw [mem_preimage, f.map_add]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : OrderedSemiring 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : AddCommMonoid G
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : Module 𝕜 G
      f : LinearMap (RingHom.id 𝕜) E F
      S : ConvexCone 𝕜 F
      x : E
      hx : Membership.mem (Set.preimage ⇑f ↑S) x
      y : E
      hy : Membership.mem (Set.preimage ⇑f ↑S) y
      ⊢ Membership.mem (↑S) (HAdd.hAdd (f x) (f y))
    -/
    exact S.add_mem hx hy
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comap (f : E →ₗ[𝕜] F) (S : ConvexCone 𝕜 F) : (S.comap f : Set E) = f ⁻¹' S :=
  rfl


@[simp] -- Porting note: was not a `dsimp` lemma
theorem comap_id (S : ConvexCone 𝕜 E) : S.comap LinearMap.id = S :=
  rfl


theorem comap_comap (g : F →ₗ[𝕜] G) (f : E →ₗ[𝕜] F) (S : ConvexCone 𝕜 G) :
    (S.comap g).comap f = S.comap (g.comp f) :=
  rfl


@[simp]
theorem mem_comap {f : E →ₗ[𝕜] F} {S : ConvexCone 𝕜 F} {x : E} : x ∈ S.comap f ↔ f x ∈ S :=
  Iff.rfl


theorem smul_mem_iff {c : 𝕜} (hc : 0 < c) {x : E} : c • x ∈ S ↔ x ∈ S :=
  ⟨fun h => inv_smul_smul₀ hc.ne' x ▸ S.smul_mem (inv_pos.2 hc) h, S.smul_mem hc⟩


/-- Constructs an ordered module given an `OrderedAddCommGroup`, a cone, and a proof that
the order relation is the one defined by the cone.
-/
theorem to_orderedSMul (S : ConvexCone 𝕜 E) (h : ∀ x y : E, x ≤ y ↔ y - x ∈ S) : OrderedSMul 𝕜 E :=
  OrderedSMul.mk'
    (by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : OrderedAddCommGroup E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        h : ∀ (x y : E), Iff (LE.le x y) (Membership.mem S (HSub.hSub y x))
        ⊢ ∀ ⦃a b : E⦄ ⦃c : 𝕜⦄, LT.lt a b → LT.lt 0 c → LE.le (HSMul.hSMul c a) (HSMul. …
      -/
      intro x y z xy hz
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : OrderedAddCommGroup E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        h : ∀ (x y : E), Iff (LE.le x y) (Membership.mem S (HSub.hSub y x))
        x y : E
        z : 𝕜
        xy : LT.lt x y
        hz : LT.lt 0 z
        ⊢ LE.le (HSMul.hSMul z x) (HSMul.hSMul z y)
      -/
      rw [h (z • x) (z • y), ← smul_sub z y x]
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : OrderedAddCommGroup E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        h : ∀ (x y : E), Iff (LE.le x y) (Membership.mem S (HSub.hSub y x))
        x y : E
        z : 𝕜
        xy : LT.lt x y
        hz : LT.lt 0 z
        ⊢ Membership.mem S (HSMul.hSMul z (HSub.hSub y x))
      -/
      exact smul_mem S hz ((h x y).mp xy.le))
      /-
        🎉 no goals
      -/


/-- A convex cone is pointed if it includes `0`. -/
def Pointed (S : ConvexCone 𝕜 E) : Prop :=
  (0 : E) ∈ S


/-- A convex cone is blunt if it doesn't include `0`. -/
def Blunt (S : ConvexCone 𝕜 E) : Prop :=
  (0 : E) ∉ S


theorem pointed_iff_not_blunt (S : ConvexCone 𝕜 E) : S.Pointed ↔ ¬S.Blunt :=
  ⟨fun h₁ h₂ => h₂ h₁, Classical.not_not.mp⟩


theorem blunt_iff_not_pointed (S : ConvexCone 𝕜 E) : S.Blunt ↔ ¬S.Pointed := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    ⊢ Iff S.Blunt (Not S.Pointed)
  -/
  rw [pointed_iff_not_blunt, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem Pointed.mono {S T : ConvexCone 𝕜 E} (h : S ≤ T) : S.Pointed → T.Pointed :=
  @h _


theorem Blunt.anti {S T : ConvexCone 𝕜 E} (h : T ≤ S) : S.Blunt → T.Blunt :=
  (· ∘ @h 0)


/-- A convex cone is flat if it contains some nonzero vector `x` and its opposite `-x`. -/
def Flat : Prop :=
  ∃ x ∈ S, x ≠ (0 : E) ∧ -x ∈ S


/-- A convex cone is salient if it doesn't include `x` and `-x` for any nonzero `x`. -/
def Salient : Prop :=
  ∀ x ∈ S, x ≠ (0 : E) → -x ∉ S


theorem salient_iff_not_flat (S : ConvexCone 𝕜 E) : S.Salient ↔ ¬S.Flat := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    ⊢ Iff S.Salient (Not S.Flat)
  -/
  simp [Salient, Flat]
  /-
    🎉 no goals
  -/


theorem Flat.mono {S T : ConvexCone 𝕜 E} (h : S ≤ T) : S.Flat → T.Flat
  | ⟨x, hxS, hx, hnxS⟩ => ⟨x, h hxS, hx, h hnxS⟩


theorem Salient.anti {S T : ConvexCone 𝕜 E} (h : T ≤ S) : S.Salient → T.Salient :=
  fun hS x hxT hx hnT => hS x (h hxT) hx (h hnT)


/-- A flat cone is always pointed (contains `0`). -/
theorem Flat.pointed {S : ConvexCone 𝕜 E} (hS : S.Flat) : S.Pointed := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    hS : S.Flat
    ⊢ S.Pointed
  -/
  obtain ⟨x, hx, _, hxneg⟩ := hS
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    x : E
    hx : Membership.mem S x
    left✝ : Ne x 0
    hxneg : Membership.mem S (Neg.neg x)
    ⊢ S.Pointed
  -/
  rw [Pointed, ← add_neg_cancel x]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    x : E
    hx : Membership.mem S x
    left✝ : Ne x 0
    hxneg : Membership.mem S (Neg.neg x)
    ⊢ Membership.mem S (HAdd.hAdd x (Neg.neg x))
  -/
  exact add_mem S hx hxneg
  /-
    🎉 no goals
  -/


/-- A blunt cone (one not containing `0`) is always salient. -/
theorem Blunt.salient {S : ConvexCone 𝕜 E} : S.Blunt → S.Salient := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    ⊢ S.Blunt → S.Salient
  -/
  rw [salient_iff_not_flat, blunt_iff_not_pointed]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    S : ConvexCone 𝕜 E
    ⊢ Not S.Pointed → Not S.Flat
  -/
  exact mt Flat.pointed
  /-
    🎉 no goals
  -/


/-- A pointed convex cone defines a preorder. -/
def toPreorder (h₁ : S.Pointed) : Preorder E where
  le x y := y - x ∈ S
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    G : Type u_4
                    inst✝² : OrderedSemiring 𝕜
                    inst✝¹ : AddCommGroup E
                    inst✝ : SMul 𝕜 E
                    S : ConvexCone 𝕜 E
                    h₁ : S.Pointed
                    x : E
                    ⊢ LE.le x x
                  -/
  le_refl x := by change x - x ∈ S; rw [sub_self x]; exact h₁
                                                     /-
                                                       🎉 no goals
                                                     -/
                             /-
                               𝕜 : Type u_1
                               E : Type u_2
                               F : Type u_3
                               G : Type u_4
                               inst✝² : OrderedSemiring 𝕜
                               inst✝¹ : AddCommGroup E
                               inst✝ : SMul 𝕜 E
                               S : ConvexCone 𝕜 E
                               h₁ : S.Pointed
                               x y z : E
                               xy : LE.le x y
                               zy : LE.le y z
                               ⊢ LE.le x z
                             -/
  le_trans x y z xy zy := by simpa using add_mem S zy xy
                             /-
                               🎉 no goals
                             -/


/-- A pointed and salient cone defines a partial order. -/
def toPartialOrder (h₁ : S.Pointed) (h₂ : S.Salient) : PartialOrder E :=
  { toPreorder S h₁ with
    le_antisymm := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        ⊢ ∀ (a b : E), LE.le a b → LE.le b a → Eq a b
      -/
      intro a b ab ba
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        ab : LE.le a b
        ba : LE.le b a
        ⊢ Eq a b
      -/
      by_contra h
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        ab : LE.le a b
        ba : LE.le b a
        h : Not (Eq a b)
        ⊢ False
      -/
      have h' : b - a ≠ 0 := fun h'' => h (eq_of_sub_eq_zero h'').symm
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        ab : LE.le a b
        ba : LE.le b a
        h : Not (Eq a b)
        h' : Ne (HSub.hSub b a) 0
        ⊢ False
      -/
      have H := h₂ (b - a) ab h'
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        ab : LE.le a b
        ba : LE.le b a
        h : Not (Eq a b)
        h' : Ne (HSub.hSub b a) 0
        H : Not (Membership.mem S (Neg.neg (HSub.hSub b a)))
        ⊢ False
      -/
      rw [neg_sub b a] at H
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        ab : LE.le a b
        ba : LE.le b a
        h : Not (Eq a b)
        h' : Ne (HSub.hSub b a) 0
        H : Not (Membership.mem S (HSub.hSub a b))
        ⊢ False
      -/
      exact H ba }
      /-
        🎉 no goals
      -/


/-- A pointed and salient cone defines an `OrderedAddCommGroup`. -/
def toOrderedAddCommGroup (h₁ : S.Pointed) (h₂ : S.Salient) : OrderedAddCommGroup E :=
                                                   /-
                                                     𝕜 : Type u_1
                                                     E : Type u_2
                                                     F : Type u_3
                                                     G : Type u_4
                                                     inst✝² : OrderedSemiring 𝕜
                                                     inst✝¹ : AddCommGroup E
                                                     inst✝ : SMul 𝕜 E
                                                     S : ConvexCone 𝕜 E
                                                     h₁ : S.Pointed
                                                     h₂ : S.Salient
                                                     ⊢ AddCommGroup E
                                                   -/
  { toPartialOrder S h₁ h₂, show AddCommGroup E by infer_instance with
                                                   /-
                                                     🎉 no goals
                                                   -/
    add_le_add_left := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        ⊢ ∀ (a b : E), LE.le a b → ∀ (c : E), LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
      -/
      intro a b hab c
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        hab : LE.le a b
        c : E
        ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
      -/
      change c + b - (c + a) ∈ S
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        hab : LE.le a b
        c : E
        ⊢ Membership.mem S (HSub.hSub (HAdd.hAdd c b) (HAdd.hAdd c a))
      -/
      rw [add_sub_add_left_eq_sub]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : SMul 𝕜 E
        S : ConvexCone 𝕜 E
        h₁ : S.Pointed
        h₂ : S.Salient
        a b : E
        hab : LE.le a b
        c : E
        ⊢ Membership.mem S (HSub.hSub b a)
      -/
      exact hab }
      /-
        🎉 no goals
      -/


instance : Zero (ConvexCone 𝕜 E) :=
                     /-
                       𝕜 : Type u_1
                       E : Type u_2
                       F : Type u_3
                       G : Type u_4
                       inst✝² : OrderedSemiring 𝕜
                       inst✝¹ : AddCommMonoid E
                       inst✝ : Module 𝕜 E
                       x✝¹ : 𝕜
                       x✝ : LT.lt 0 x✝¹
                       ⊢ ∀ ⦃x : E⦄, Membership.mem 0 x → Membership.mem 0 (HSMul.hSMul x✝¹ x)
                     -/
                     /-
                       🎉 no goals
                     -/
  ⟨⟨0, fun _ _ => by simp, fun _ => by simp⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem mem_zero (x : E) : x ∈ (0 : ConvexCone 𝕜 E) ↔ x = 0 :=
  Iff.rfl


@[simp]
theorem coe_zero : ((0 : ConvexCone 𝕜 E) : Set E) = 0 :=
  rfl


                                                          /-
                                                            𝕜 : Type u_1
                                                            E : Type u_2
                                                            inst✝² : OrderedSemiring 𝕜
                                                            inst✝¹ : AddCommMonoid E
                                                            inst✝ : Module 𝕜 E
                                                            ⊢ ConvexCone.Pointed 0
                                                          -/
theorem pointed_zero : (0 : ConvexCone 𝕜 E).Pointed := by rw [Pointed, mem_zero]
                                                          /-
                                                            🎉 no goals
                                                          -/


instance instAdd : Add (ConvexCone 𝕜 E) :=
  ⟨fun K₁ K₂ =>
    { carrier := { z | ∃ x ∈ K₁, ∃ y ∈ K₂, x + y = z }
      smul_mem' := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          ⊢ ∀ ⦃c : 𝕜⦄, LT.lt 0 c → ∀ ⦃x : E⦄, Membership.mem (setOf fun z => Exists fun  …
        -/
        rintro c hc _ ⟨x, hx, y, hy, rfl⟩
        /-
          case intro.intro.intro.intro
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          c : 𝕜
          hc : LT.lt 0 c
          x : E
          hx : Membership.mem K₁ x
          y : E
          hy : Membership.mem K₂ y
          ⊢ Membership.mem (setOf fun z => Exists fun x => And (Membership.mem K₁ x) (Ex …
        -/
        rw [smul_add]
        /-
          case intro.intro.intro.intro
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          c : 𝕜
          hc : LT.lt 0 c
          x : E
          hx : Membership.mem K₁ x
          y : E
          hy : Membership.mem K₂ y
          ⊢ Membership.mem (setOf fun z => Exists fun x => And (Membership.mem K₁ x) (Ex …
        -/
        use c • x, K₁.smul_mem hc hx, c • y, K₂.smul_mem hc hy
        /-
          🎉 no goals
        -/
      add_mem' := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          ⊢ ∀ ⦃x : E⦄, Membership.mem (setOf fun z => Exists fun x => And (Membership.me …
        -/
        rintro _ ⟨x₁, hx₁, x₂, hx₂, rfl⟩ y ⟨y₁, hy₁, y₂, hy₂, rfl⟩
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          x₁ : E
          hx₁ : Membership.mem K₁ x₁
          x₂ : E
          hx₂ : Membership.mem K₂ x₂
          y₁ : E
          hy₁ : Membership.mem K₁ y₁
          y₂ : E
          hy₂ : Membership.mem K₂ y₂
          ⊢ Membership.mem (setOf fun z => Exists fun x => And (Membership.mem K₁ x) (Ex …
        -/
        use x₁ + y₁, K₁.add_mem hx₁ hy₁, x₂ + y₂, K₂.add_mem hx₂ hy₂
        /-
          case right
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          K₁ K₂ : ConvexCone 𝕜 E
          x₁ : E
          hx₁ : Membership.mem K₁ x₁
          x₂ : E
          hx₂ : Membership.mem K₂ x₂
          y₁ : E
          hy₁ : Membership.mem K₁ y₁
          y₂ : E
          hy₂ : Membership.mem K₂ y₂
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd x₁ y₁) (HAdd.hAdd x₂ y₂)) (HAdd.hAdd (HAdd.hAdd x₁  …
        -/
        /-
          🎉 no goals
        -/
        abel }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem mem_add {K₁ K₂ : ConvexCone 𝕜 E} {a : E} :
    a ∈ K₁ + K₂ ↔ ∃ x ∈ K₁, ∃ y ∈ K₂, x + y = a :=
  Iff.rfl


instance instAddZeroClass : AddZeroClass (ConvexCone 𝕜 E) where
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     F : Type u_3
                     G : Type u_4
                     inst✝² : OrderedSemiring 𝕜
                     inst✝¹ : AddCommMonoid E
                     inst✝ : Module 𝕜 E
                     x✝ : ConvexCone 𝕜 E
                     ⊢ Eq (HAdd.hAdd 0 x✝) x✝
                   -/
  zero_add _ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     F : Type u_3
                     G : Type u_4
                     inst✝² : OrderedSemiring 𝕜
                     inst✝¹ : AddCommMonoid E
                     inst✝ : Module 𝕜 E
                     x✝ : ConvexCone 𝕜 E
                     ⊢ Eq (HAdd.hAdd x✝ 0) x✝
                   -/
  add_zero _ := by ext; simp
                        /-
                          🎉 no goals
                        -/


instance instAddCommSemigroup : AddCommSemigroup (ConvexCone 𝕜 E) where
  add := Add.add
  add_assoc _ _ _ := SetLike.coe_injective <| add_assoc _ _ _
  add_comm _ _ := SetLike.coe_injective <| add_comm _ _


/-- Every submodule is trivially a convex cone. -/
def toConvexCone (S : Submodule 𝕜 E) : ConvexCone 𝕜 E where
  carrier := S
  smul_mem' c _ _ hx := S.smul_mem c hx
  add_mem' _ hx _ hy := S.add_mem hx hy


@[simp]
theorem coe_toConvexCone (S : Submodule 𝕜 E) : ↑S.toConvexCone = (S : Set E) :=
  rfl


@[simp]
theorem mem_toConvexCone {x : E} {S : Submodule 𝕜 E} : x ∈ S.toConvexCone ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem toConvexCone_le_iff {S T : Submodule 𝕜 E} : S.toConvexCone ≤ T.toConvexCone ↔ S ≤ T :=
  Iff.rfl


@[simp]
theorem toConvexCone_bot : (⊥ : Submodule 𝕜 E).toConvexCone = 0 :=
  rfl


@[simp]
theorem toConvexCone_top : (⊤ : Submodule 𝕜 E).toConvexCone = ⊤ :=
  rfl


@[simp]
theorem toConvexCone_inf (S T : Submodule 𝕜 E) :
    (S ⊓ T).toConvexCone = S.toConvexCone ⊓ T.toConvexCone :=
  rfl


@[simp]
theorem pointed_toConvexCone (S : Submodule 𝕜 E) : S.toConvexCone.Pointed :=
  S.zero_mem


/-- The positive cone is the convex cone formed by the set of nonnegative elements in an ordered
module.
-/
def positive : ConvexCone 𝕜 E where
  carrier := Set.Ici 0
  smul_mem' _ hc _ (hx : _ ≤ _) := smul_nonneg hc.le hx
  add_mem' _ (hx : _ ≤ _) _ (hy : _ ≤ _) := add_nonneg hx hy


@[simp]
theorem mem_positive {x : E} : x ∈ positive 𝕜 E ↔ 0 ≤ x :=
  Iff.rfl


@[simp]
theorem coe_positive : ↑(positive 𝕜 E) = Set.Ici (0 : E) :=
  rfl


/-- The positive cone of an ordered module is always salient. -/
theorem salient_positive : Salient (positive 𝕜 E) := fun x xs hx hx' =>
  lt_irrefl (0 : E)
    (calc
      0 < x := lt_of_le_of_ne xs hx.symm
      _ ≤ x + -x := le_add_of_nonneg_right hx'
      _ = 0 := add_neg_cancel x
      )


/-- The positive cone of an ordered module is always pointed. -/
theorem pointed_positive : Pointed (positive 𝕜 E) :=
  le_refl 0


/-- The cone of strictly positive elements.

Note that this naming diverges from the mathlib convention of `pos` and `nonneg` due to "positive
cone" (`ConvexCone.positive`) being established terminology for the non-negative elements. -/
def strictlyPositive : ConvexCone 𝕜 E where
  carrier := Set.Ioi 0
  smul_mem' _ hc _ (hx : _ < _) := smul_pos hc hx
  add_mem' _ hx _ hy := add_pos hx hy


@[simp]
theorem mem_strictlyPositive {x : E} : x ∈ strictlyPositive 𝕜 E ↔ 0 < x :=
  Iff.rfl


@[simp]
theorem coe_strictlyPositive : ↑(strictlyPositive 𝕜 E) = Set.Ioi (0 : E) :=
  rfl


theorem positive_le_strictlyPositive : strictlyPositive 𝕜 E ≤ positive 𝕜 E := fun _ => le_of_lt


/-- The strictly positive cone of an ordered module is always salient. -/
theorem salient_strictlyPositive : Salient (strictlyPositive 𝕜 E) :=
  (salient_positive 𝕜 E).anti <| positive_le_strictlyPositive 𝕜 E


/-- The strictly positive cone of an ordered module is always blunt. -/
theorem blunt_strictlyPositive : Blunt (strictlyPositive 𝕜 E) :=
  lt_irrefl 0


/-- The set of vectors proportional to those in a convex set forms a convex cone. -/
def toCone (s : Set E) (hs : Convex 𝕜 s) : ConvexCone 𝕜 E := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    ⊢ ConvexCone 𝕜 E
  -/
  apply ConvexCone.mk (⋃ (c : 𝕜) (_ : 0 < c), c • s) <;> simp only [mem_iUnion, mem_smul_set]
    /-
      case smul_mem'
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      ⊢ ∀ ⦃c : 𝕜⦄, LT.lt 0 c → ∀ ⦃x : E⦄, (Exists fun i => Exists fun h => Exists fu …
    -/
  · rintro c c_pos _ ⟨c', c'_pos, x, hx, rfl⟩
    /-
      case smul_mem'.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      c : 𝕜
      c_pos : LT.lt 0 c
      c' : 𝕜
      c'_pos : LT.lt 0 c'
      x : E
      hx : Membership.mem s x
      ⊢ Exists fun i => Exists fun h => Exists fun y => And (Membership.mem s y) (Eq …
    -/
    exact ⟨c * c', mul_pos c_pos c'_pos, x, hx, (smul_smul _ _ _).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case add_mem'
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      ⊢ ∀ ⦃x : E⦄, (Exists fun i => Exists fun h => Exists fun y => And (Membership. …
    -/
  · rintro _ ⟨cx, cx_pos, x, hx, rfl⟩ _ ⟨cy, cy_pos, y, hy, rfl⟩
    /-
      case add_mem'.intro.intro.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      cx : 𝕜
      cx_pos : LT.lt 0 cx
      x : E
      hx : Membership.mem s x
      cy : 𝕜
      cy_pos : LT.lt 0 cy
      y : E
      hy : Membership.mem s y
      ⊢ Exists fun i => Exists fun h => Exists fun y_1 => And (Membership.mem s y_1) …
    -/
    have : 0 < cx + cy := add_pos cx_pos cy_pos
    /-
      case add_mem'.intro.intro.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      cx : 𝕜
      cx_pos : LT.lt 0 cx
      x : E
      hx : Membership.mem s x
      cy : 𝕜
      cy_pos : LT.lt 0 cy
      y : E
      hy : Membership.mem s y
      this : LT.lt 0 (HAdd.hAdd cx cy)
      ⊢ Exists fun i => Exists fun h => Exists fun y_1 => And (Membership.mem s y_1) …
    -/
    refine ⟨_, this, _, convex_iff_div.1 hs hx hy cx_pos.le cy_pos.le this, ?_⟩
    /-
      case add_mem'.intro.intro.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      cx : 𝕜
      cx_pos : LT.lt 0 cx
      x : E
      hx : Membership.mem s x
      cy : 𝕜
      cy_pos : LT.lt 0 cy
      y : E
      hy : Membership.mem s y
      this : LT.lt 0 (HAdd.hAdd cx cy)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd cx cy) (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv cx (HAd …
    -/
    simp only [smul_add, smul_smul, mul_div_assoc', mul_div_cancel_left₀ _ this.ne']
    /-
      🎉 no goals
    -/


theorem mem_toCone : x ∈ hs.toCone s ↔ ∃ c : 𝕜, 0 < c ∧ ∃ y ∈ s, c • y = x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x : E
    ⊢ Iff (Membership.mem (Convex.toCone s hs) x) (Exists fun c => And (LT.lt 0 c) …
  -/
  simp only [toCone, ConvexCone.mem_mk, mem_iUnion, mem_smul_set, eq_comm, exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_toCone' : x ∈ hs.toCone s ↔ ∃ c : 𝕜, 0 < c ∧ c • x ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x : E
    ⊢ Iff (Membership.mem (Convex.toCone s hs) x) (Exists fun c => And (LT.lt 0 c) …
  -/
  refine hs.mem_toCone.trans ⟨?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      ⊢ (Exists fun c => And (LT.lt 0 c) (Exists fun y => And (Membership.mem s y) ( …
    -/
  · rintro ⟨c, hc, y, hy, rfl⟩
    /-
      case refine_1.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      c : 𝕜
      hc : LT.lt 0 c
      y : E
      hy : Membership.mem s y
      ⊢ Exists fun c_1 => And (LT.lt 0 c_1) (Membership.mem s (HSMul.hSMul c_1 (HSMu …
    -/
    exact ⟨c⁻¹, inv_pos.2 hc, by rwa [smul_smul, inv_mul_cancel₀ hc.ne', one_smul]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      ⊢ (Exists fun c => And (LT.lt 0 c) (Membership.mem s (HSMul.hSMul c x))) → Exi …
    -/
  · rintro ⟨c, hc, hcx⟩
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      x : E
      c : 𝕜
      hc : LT.lt 0 c
      hcx : Membership.mem s (HSMul.hSMul c x)
      ⊢ Exists fun c => And (LT.lt 0 c) (Exists fun y => And (Membership.mem s y) (E …
    -/
    exact ⟨c⁻¹, inv_pos.2 hc, _, hcx, by rw [smul_smul, inv_mul_cancel₀ hc.ne', one_smul]⟩
    /-
      🎉 no goals
    -/


theorem subset_toCone : s ⊆ hs.toCone s := fun x hx =>
                                       /-
                                         𝕜 : Type u_1
                                         E : Type u_2
                                         inst✝² : LinearOrderedField 𝕜
                                         inst✝¹ : AddCommGroup E
                                         inst✝ : Module 𝕜 E
                                         s : Set E
                                         hs : Convex 𝕜 s
                                         x : E
                                         hx : Membership.mem s x
                                         ⊢ Membership.mem s (HSMul.hSMul 1 x)
                                       -/
  hs.mem_toCone'.2 ⟨1, zero_lt_one, by rwa [one_smul]⟩
                                       /-
                                         🎉 no goals
                                       -/


/-- `hs.toCone s` is the least cone that includes `s`. -/
theorem toCone_isLeast : IsLeast { t : ConvexCone 𝕜 E | s ⊆ t } (hs.toCone s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    ⊢ IsLeast (setOf fun t => HasSubset.Subset s ↑t) (Convex.toCone s hs)
  -/
  refine ⟨hs.subset_toCone, fun t ht x hx => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    t : ConvexCone 𝕜 E
    ht : Membership.mem (setOf fun t => HasSubset.Subset s ↑t) t
    x : E
    hx : Membership.mem (Convex.toCone s hs) x
    ⊢ Membership.mem t x
  -/
  rcases hs.mem_toCone.1 hx with ⟨c, hc, y, hy, rfl⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    t : ConvexCone 𝕜 E
    ht : Membership.mem (setOf fun t => HasSubset.Subset s ↑t) t
    c : 𝕜
    hc : LT.lt 0 c
    y : E
    hy : Membership.mem s y
    hx : Membership.mem (Convex.toCone s hs) (HSMul.hSMul c y)
    ⊢ Membership.mem t (HSMul.hSMul c y)
  -/
  exact t.smul_mem hc (ht hy)
  /-
    🎉 no goals
  -/


theorem toCone_eq_sInf : hs.toCone s = sInf { t : ConvexCone 𝕜 E | s ⊆ t } :=
  hs.toCone_isLeast.isGLB.sInf_eq.symm


theorem convexHull_toCone_isLeast (s : Set E) :
    IsLeast { t : ConvexCone 𝕜 E | s ⊆ t } ((convex_convexHull 𝕜 s).toCone _) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ IsLeast (setOf fun t => HasSubset.Subset s ↑t) (Convex.toCone ((convexHull 𝕜 …
  -/
  convert (convex_convexHull 𝕜 s).toCone_isLeast using 1
  /-
    case h.e'_3
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq (setOf fun t => HasSubset.Subset s ↑t) (setOf fun t => HasSubset.Subset ( …
  -/
  ext t
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    t : ConvexCone 𝕜 E
    ⊢ Iff (Membership.mem (setOf fun t => HasSubset.Subset s ↑t) t) (Membership.me …
  -/
  exact ⟨fun h => convexHull_min h t.convex, (subset_convexHull 𝕜 s).trans⟩
  /-
    🎉 no goals
  -/


theorem convexHull_toCone_eq_sInf (s : Set E) :
    (convex_convexHull 𝕜 s).toCone _ = sInf { t : ConvexCone 𝕜 E | s ⊆ t } :=
  Eq.symm <| IsGLB.sInf_eq <| IsLeast.isGLB <| convexHull_toCone_isLeast s



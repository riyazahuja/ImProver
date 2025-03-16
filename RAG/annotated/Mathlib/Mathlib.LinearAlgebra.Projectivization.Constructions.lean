/-- Orthogonality on the projective plane. -/
def orthogonal : ℙ F (m → F) → ℙ F (m → F) → Prop :=
  Quotient.lift₂ (fun v w ↦ dotProduct v.1 w.1 = 0) (fun _ _ _ _ ⟨_, h1⟩ ⟨_, h2⟩ ↦ by
    simp_rw [← h1, ← h2, dotProduct_smul, smul_dotProduct, smul_smul,
      smul_eq_zero_iff_eq])


lemma orthogonal_mk {v w : m → F} (hv : v ≠ 0) (hw : w ≠ 0) :
    orthogonal (mk F v hv) (mk F w hw) ↔ dotProduct v w = 0 :=
  Iff.rfl


lemma orthogonal_comm {v w : ℙ F (m → F)} : orthogonal v w ↔ orthogonal w v := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v w : Projectivization F (m → F)
    ⊢ Iff (v.orthogonal w) (w.orthogonal v)
  -/
  induction' v with v hv
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    w : Projectivization F (m → F)
    v : m → F
    hv : Ne v 0
    ⊢ Iff ((Projectivization.mk F v hv).orthogonal w) (w.orthogonal (Projectivizat …
  -/
  induction' w with w hw
  /-
    case h.h
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : m → F
    hv : Ne v 0
    w : m → F
    hw : Ne w 0
    ⊢ Iff ((Projectivization.mk F v hv).orthogonal (Projectivization.mk F w hw)) ( …
  -/
  rw [orthogonal_mk hv hw, orthogonal_mk hw hv, dotProduct_comm]
  /-
    🎉 no goals
  -/


lemma exists_not_self_orthogonal (v : ℙ F (m → F)) : ∃ w, ¬ orthogonal v w := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : Projectivization F (m → F)
    ⊢ Exists fun w => Not (v.orthogonal w)
  -/
  induction' v with v hv
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : m → F
    hv : Ne v 0
    ⊢ Exists fun w => Not ((Projectivization.mk F v hv).orthogonal w)
  -/
  rw [ne_eq, ← dotProduct_eq_zero_iff, not_forall] at hv
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : m → F
    hv✝ : Ne v 0
    hv : Exists fun x => Not (Eq (dotProduct v x) 0)
    ⊢ Exists fun w => Not ((Projectivization.mk F v hv✝).orthogonal w)
  -/
  obtain ⟨w, hw⟩ := hv
  /-
    case h.intro
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : m → F
    hv : Ne v 0
    w : m → F
    hw : Not (Eq (dotProduct v w) 0)
    ⊢ Exists fun w => Not ((Projectivization.mk F v hv).orthogonal w)
  -/
  exact ⟨mk F w fun h ↦ hw (by rw [h, dotProduct_zero]), hw⟩
  /-
    🎉 no goals
  -/


lemma exists_not_orthogonal_self (v : ℙ F (m → F)) : ∃ w, ¬ orthogonal w v := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : Projectivization F (m → F)
    ⊢ Exists fun w => Not (w.orthogonal v)
  -/
  simp only [orthogonal_comm]
  /-
    F : Type u_1
    inst✝¹ : Field F
    m : Type u_2
    inst✝ : Fintype m
    v : Projectivization F (m → F)
    ⊢ Exists fun w => Not (v.orthogonal w)
  -/
  exact exists_not_self_orthogonal v
  /-
    🎉 no goals
  -/


lemma mk_eq_mk_iff_crossProduct_eq_zero {v w : Fin 3 → F} (hv : v ≠ 0) (hw : w ≠ 0) :
    mk F v hv = mk F w hw ↔ crossProduct v w = 0 := by
  rw [← not_iff_not, mk_eq_mk_iff', not_exists, ← LinearIndependent.pair_iff' hw,
    ← crossProduct_ne_zero_iff_linearIndependent, ← cross_anticomm, neg_ne_zero]


/-- Cross product on the projective plane. -/
def cross : ℙ F (Fin 3 → F) → ℙ F (Fin 3 → F) → ℙ F (Fin 3 → F) :=
  Quotient.map₂ (fun v w ↦ if h : crossProduct v.1 w.1 = 0 then v else ⟨crossProduct v.1 w.1, h⟩)
    (fun _ _ ⟨a, ha⟩ _ _ ⟨b, hb⟩ ↦ by
      simp_rw [← ha, ← hb, LinearMap.map_smul_of_tower, LinearMap.smul_apply, smul_smul,
        mul_comm b a, smul_eq_zero_iff_eq]
      /-
        F : Type u_1
        inst✝² : Field F
        m : Type u_2
        inst✝¹ : Fintype m
        inst✝ : DecidableEq F
        x✝⁵ x✝⁴ : Subtype fun v => Ne v 0
        x✝³ : HasEquiv.Equiv x✝⁵ x✝⁴
        x✝² x✝¹ : Subtype fun v => Ne v 0
        x✝ : HasEquiv.Equiv x✝² x✝¹
        a : Units F
        ha : Eq ((fun m => HSMul.hSMul m ↑x✝⁴) a) ↑x✝⁵
        b : Units F
        hb : Eq ((fun m => HSMul.hSMul m ↑x✝¹) b) ↑x✝²
        ⊢ HasEquiv.Equiv (dite (Eq ((crossProduct ↑x✝⁴) ↑x✝¹) 0) (fun h => x✝⁵) fun h  …
      -/
      split_ifs
        /-
          case pos
          F : Type u_1
          inst✝² : Field F
          m : Type u_2
          inst✝¹ : Fintype m
          inst✝ : DecidableEq F
          x✝⁵ x✝⁴ : Subtype fun v => Ne v 0
          x✝³ : HasEquiv.Equiv x✝⁵ x✝⁴
          x✝² x✝¹ : Subtype fun v => Ne v 0
          x✝ : HasEquiv.Equiv x✝² x✝¹
          a : Units F
          ha : Eq ((fun m => HSMul.hSMul m ↑x✝⁴) a) ↑x✝⁵
          b : Units F
          hb : Eq ((fun m => HSMul.hSMul m ↑x✝¹) b) ↑x✝²
          h✝ : Eq ((crossProduct ↑x✝⁴) ↑x✝¹) 0
          ⊢ HasEquiv.Equiv x✝⁵ x✝⁴
        -/
      · use a
        /-
          🎉 no goals
        -/
        /-
          case neg
          F : Type u_1
          inst✝² : Field F
          m : Type u_2
          inst✝¹ : Fintype m
          inst✝ : DecidableEq F
          x✝⁵ x✝⁴ : Subtype fun v => Ne v 0
          x✝³ : HasEquiv.Equiv x✝⁵ x✝⁴
          x✝² x✝¹ : Subtype fun v => Ne v 0
          x✝ : HasEquiv.Equiv x✝² x✝¹
          a : Units F
          ha : Eq ((fun m => HSMul.hSMul m ↑x✝⁴) a) ↑x✝⁵
          b : Units F
          hb : Eq ((fun m => HSMul.hSMul m ↑x✝¹) b) ↑x✝²
          h✝ : Not (Eq ((crossProduct ↑x✝⁴) ↑x✝¹) 0)
          ⊢ HasEquiv.Equiv ⟨HSMul.hSMul (HMul.hMul a b) ((crossProduct ↑x✝⁴) ↑x✝¹), ⋯⟩ ⟨ …
        -/
      · use a * b)
        /-
          🎉 no goals
        -/


lemma cross_mk {v w : Fin 3 → F} (hv : v ≠ 0) (hw : w ≠ 0) :
    cross (mk F v hv) (mk F w hw) =
      if h : crossProduct v w = 0 then mk F v hv else mk F (crossProduct v w) h := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Fin 3 → F
    hv : Ne v 0
    hw : Ne w 0
    ⊢ Eq ((Projectivization.mk F v hv).cross (Projectivization.mk F w hw)) (dite ( …
  -/
  change Quotient.mk'' _ = _
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Fin 3 → F
    hv : Ne v 0
    hw : Ne w 0
    ⊢ Eq (Quotient.mk'' ((fun v w => dite (Eq ((crossProduct ↑v) ↑w) 0) (fun h =>  …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  split_ifs with h <;> simp only [h] <;> rfl
                                         /-
                                           🎉 no goals
                                         -/


lemma cross_mk_of_cross_eq_zero {v w : Fin 3 → F} (hv : v ≠ 0) (hw : w ≠ 0)
    (h : crossProduct v w = 0) :
    cross (mk F v hv) (mk F w hw) = mk F v hv := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Fin 3 → F
    hv : Ne v 0
    hw : Ne w 0
    h : Eq ((crossProduct v) w) 0
    ⊢ Eq ((Projectivization.mk F v hv).cross (Projectivization.mk F w hw)) (Projec …
  -/
  rw [cross_mk, dif_pos h]
  /-
    🎉 no goals
  -/


lemma cross_mk_of_cross_ne_zero {v w : Fin 3 → F} (hv : v ≠ 0) (hw : w ≠ 0)
    (h : crossProduct v w ≠ 0) :
    cross (mk F v hv) (mk F w hw) = mk F (crossProduct v w) h := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Fin 3 → F
    hv : Ne v 0
    hw : Ne w 0
    h : Ne ((crossProduct v) w) 0
    ⊢ Eq ((Projectivization.mk F v hv).cross (Projectivization.mk F w hw)) (Projec …
  -/
  rw [cross_mk, dif_neg h]
  /-
    🎉 no goals
  -/


lemma cross_self (v : ℙ F (Fin 3 → F)) : cross v v = v := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v : Projectivization F (Fin 3 → F)
    ⊢ Eq (v.cross v) v
  -/
  induction' v with v hv
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v : Fin 3 → F
    hv : Ne v 0
    ⊢ Eq ((Projectivization.mk F v hv).cross (Projectivization.mk F v hv)) (Projec …
  -/
  rw [cross_mk_of_cross_eq_zero]
  /-
    case h.h
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v : Fin 3 → F
    hv : Ne v 0
    ⊢ Eq ((crossProduct v) v) 0
  -/
  rw [← mk_eq_mk_iff_crossProduct_eq_zero hv]
  /-
    🎉 no goals
  -/


lemma cross_mk_of_ne {v w : Fin 3 → F} (hv : v ≠ 0) (hw : w ≠ 0) (h : mk F v hv ≠ mk F w hw) :
    cross (mk F v hv) (mk F w hw) = mk F (crossProduct v w)
      (mt (mk_eq_mk_iff_crossProduct_eq_zero hv hw).mpr h) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Fin 3 → F
    hv : Ne v 0
    hw : Ne w 0
    h : Ne (Projectivization.mk F v hv) (Projectivization.mk F w hw)
    ⊢ Eq ((Projectivization.mk F v hv).cross (Projectivization.mk F w hw)) (Projec …
  -/
  rw [cross_mk_of_cross_ne_zero]
  /-
    🎉 no goals
  -/


lemma cross_comm (v w : ℙ F (Fin 3 → F)) : cross v w = cross w v := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    ⊢ Eq (v.cross w) (w.cross v)
  -/
  rcases eq_or_ne v w with rfl | h
    /-
      case inl
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : DecidableEq F
      v : Projectivization F (Fin 3 → F)
      ⊢ Eq (v.cross v) (v.cross v)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : DecidableEq F
      v w : Projectivization F (Fin 3 → F)
      h : Ne v w
      ⊢ Eq (v.cross w) (w.cross v)
    -/
  · induction' v with v hv
    /-
      case inr.h
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : DecidableEq F
      w : Projectivization F (Fin 3 → F)
      v : Fin 3 → F
      hv : Ne v 0
      h : Ne (Projectivization.mk F v hv) w
      ⊢ Eq ((Projectivization.mk F v hv).cross w) (w.cross (Projectivization.mk F v  …
    -/
    induction' w with w hw
    rw [cross_mk_of_ne hv hw h, cross_mk_of_ne hw hv h.symm, mk_eq_mk_iff_crossProduct_eq_zero,
      ← cross_anticomm v w, map_neg, _root_.cross_self, neg_zero]


theorem cross_orthogonal_left {v w : ℙ F (Fin 3 → F)} (h : v ≠ w) :
    (cross v w).orthogonal v := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ (v.cross w).orthogonal v
  -/
  induction' v with v hv
  /-
    case h
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    w : Projectivization F (Fin 3 → F)
    v : Fin 3 → F
    hv : Ne v 0
    h : Ne (Projectivization.mk F v hv) w
    ⊢ ((Projectivization.mk F v hv).cross w).orthogonal (Projectivization.mk F v hv)
  -/
  induction' w with w hw
  /-
    case h.h
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v : Fin 3 → F
    hv : Ne v 0
    w : Fin 3 → F
    hw : Ne w 0
    h : Ne (Projectivization.mk F v hv) (Projectivization.mk F w hw)
    ⊢ ((Projectivization.mk F v hv).cross (Projectivization.mk F w hw)).orthogonal …
  -/
  rw [cross_mk_of_ne hv hw h, orthogonal_mk, dotProduct_comm, dot_self_cross]
  /-
    🎉 no goals
  -/


theorem cross_orthogonal_right {v w : ℙ F (Fin 3 → F)} (h : v ≠ w) :
    (cross v w).orthogonal w := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ (v.cross w).orthogonal w
  -/
  rw [cross_comm]
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ (w.cross v).orthogonal w
  -/
  exact cross_orthogonal_left h.symm
  /-
    🎉 no goals
  -/


theorem orthogonal_cross_left {v w : ℙ F (Fin 3 → F)} (h : v ≠ w) :
    v.orthogonal (cross v w) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ v.orthogonal (v.cross w)
  -/
  rw [orthogonal_comm]
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ (v.cross w).orthogonal v
  -/
  exact cross_orthogonal_left h
  /-
    🎉 no goals
  -/


lemma orthogonal_cross_right {v w : ℙ F (Fin 3 → F)} (h : v ≠ w) :
    w.orthogonal (cross v w) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ w.orthogonal (v.cross w)
  -/
  rw [orthogonal_comm]
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    v w : Projectivization F (Fin 3 → F)
    h : Ne v w
    ⊢ (v.cross w).orthogonal w
  -/
  exact cross_orthogonal_right h
  /-
    🎉 no goals
  -/


